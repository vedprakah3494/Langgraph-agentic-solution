import json
import os
import re
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import OpenAI

_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
for _key in (
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_MODEL",
    "AZURE_DEVOPS_ORGANIZATION",
    "AZURE_DEVOPS_PROJECT",
    "AZURE_DEVOPS_PAT",
):
    if _key in os.environ and not os.environ[_key].strip():
        del os.environ[_key]
load_dotenv(dotenv_path=_DOTENV_PATH)

from agent.azure_boards import (
    create_user_story,
    get_user_story,
    update_user_story,
    validate_configuration,
)


class AgentState(TypedDict, total=False):
    request: str
    plan: dict
    mode: str
    work_item_id: int
    story: dict
    result: str


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _client() -> OpenAI:
    return OpenAI(
        base_url=_required_env("OLLAMA_BASE_URL"),
        api_key=_required_env("OLLAMA_API_KEY"),
    )


def _decode_json_response(content: str) -> dict:
    candidate = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The model returned an invalid work-item plan.") from exc


def _ask_for_plan(client: OpenAI, request: str, *, create_only: bool = False) -> dict:
    instruction = (
        "You help manage Azure Boards user stories. Return exactly one valid JSON "
        "object without Markdown fences or other text. Return only a JSON object "
        "with keys action, work_item_id, title, description, acceptance_criteria, "
        "clarification. action must be create, update, or clarify. For updates, "
        "work_item_id must be an integer. For creates, write a concise title and "
        "a description in 'As a ..., I want ..., so that ...' form only when the "
        "request supports those details; otherwise ask a clarification. Do not "
        "invent requirements. Preserve only requested changes for updates; set "
        "unspecified fields to null. Never choose update without an existing "
        "Azure Boards work item ID."
    )
    if create_only:
        instruction += (
            " The request is for a NEW user story, not an update. You must set "
            "action to create. Interpret the supplied user request as the desired "
            "story behavior. Do not ask for a work item ID."
        )
    response = client.chat.completions.create(
        model=_required_env("OLLAMA_MODEL"),
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": request},
        ],
    )
    try:
        content = response.choices[0].message.content or ""
    except IndexError as exc:
        raise RuntimeError("The model returned no choices.") from exc
    return _decode_json_response(content)


def _parse_request(state: AgentState) -> AgentState:
    client = _client()
    request = state["request"]
    plan = _ask_for_plan(client, request)

    action = plan.get("action")
    if action not in {"create", "update", "clarify"}:
        raise RuntimeError(f"The model returned an unsupported action: {action!r}")
    explicit_update = bool(
        re.search(r"\b(update|edit|modify|change|revise)\b", request, re.IGNORECASE)
    )
    ids_in_request = re.findall(r"(?<!\w)#?(\d{1,10})(?!\w)", request)
    asks_for_update_id = (
        action == "clarify"
        and "work item" in str(plan.get("clarification", "")).lower()
        and "id" in str(plan.get("clarification", "")).lower()
    )
    if (action == "update" or asks_for_update_id) and not explicit_update:
        plan = _ask_for_plan(client, request, create_only=True)
        action = plan.get("action")
        if action not in {"create", "update", "clarify"}:
            raise RuntimeError(f"The model returned an unsupported action: {action!r}")
    if action == "update":
        if not explicit_update or not ids_in_request:
            plan = {
                "action": "clarify",
                "clarification": (
                    "To update a story, say what to change and include its Azure "
                    "Boards work item ID. Otherwise describe a new story to create."
                ),
            }
        else:
            plan["work_item_id"] = int(ids_in_request[0])
    if action == "create" and not (plan.get("title") and plan.get("description")):
        plan = {
            "action": "clarify",
            "clarification": "Please provide enough detail for a story title and description.",
        }
    if action == "update":
        has_changes = any(
            plan.get(key) is not None
            for key in ("title", "description", "acceptance_criteria")
        )
        if not isinstance(plan.get("work_item_id"), int) or not has_changes:
            plan = {
                "action": "clarify",
                "clarification": (
                    "Please provide the Azure Boards work item ID and at least "
                    "one field to update."
                ),
            }
    return {"plan": plan}


def _chat_user_text(messages: list[dict]) -> list[str]:
    return [
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]


def _user_acceptance_criteria(messages: list[dict]) -> str | None:
    previous_assistant = ""
    for message in messages:
        content = str(message.get("content", "")).strip()
        if message.get("role") == "assistant":
            previous_assistant = content.lower()
            continue
        if message.get("role") != "user" or not content:
            continue

        explicit = re.search(
            r"(?:acceptance\s+criteria|acceptance\s+criterion|test\s+cases?)\s*[:\-]\s*(.+)",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if explicit:
            return explicit.group(1).strip()
        if re.search(r"\bgiven\b.+\bwhen\b.+\bthen\b", content, re.IGNORECASE | re.DOTALL):
            return content
        if any(
            term in previous_assistant
            for term in ("acceptance criteria", "acceptance criterion", "testable behavior")
        ):
            return content
    return None


def _has_stated_benefit(messages: list[dict]) -> bool:
    text = " ".join(_chat_user_text(messages)).lower()
    return bool(
        re.search(
            r"\bso that\b|\bbecause\b|\bin order to\b|\bto ensure\b|"
            r"\bwhich helps\b|\bthis helps\b|\bbenefit\b|\bpurpose\b",
            text,
        )
    )


def _has_persona_and_need(messages: list[dict]) -> bool:
    text = " ".join(_chat_user_text(messages))
    return bool(
        re.search(
            r"\b[a-z][a-z0-9 -]{0,40}\s+"
            r"(?:wants?|want|needs?|need|would like to|can|must|should)\s+"
            r"(?:to\s+)?[a-z]",
            text,
            re.IGNORECASE,
        )
        or re.search(r"\bas\s+(?:a|an)\s+[a-z][a-z -]{1,40}", text, re.IGNORECASE)
    )


def prepare_chat_story(messages: list[dict]) -> dict:
    if not _has_persona_and_need(messages):
        return {
            "action": "ask",
            "reply": (
                "Who is the user, and what do they need to do? For example, "
                "'As an employee, I want to request leave.'"
            ),
        }
    acceptance_criteria = _user_acceptance_criteria(messages)
    if not acceptance_criteria:
        return {
            "action": "ask",
            "reply": (
                "What should the user be able to do, and how will we verify it? "
                "Please provide at least one testable acceptance criterion."
            ),
        }
    if not _has_stated_benefit(messages):
        return {
            "action": "ask",
            "reply": (
                "What outcome or business benefit should this provide? "
                "For example, why does this capability matter to the user?"
            ),
        }

    conversation = "\n".join(
        f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
        for message in messages
    )
    instruction = (
        "You write Azure Boards user story drafts from the conversation. All "
        "required user details (persona, need, business benefit, and testable "
        "acceptance criteria) have already been checked and provided. You must "
        "return action draft, never ask another question. Return one valid JSON "
        "object with keys action, reply, title, description, acceptance_criteria, "
        "priority, area_path, iteration_path, context. Create a concise title "
        "and a standard 'As a ..., I want ..., so that ...' description using "
        "only the user's stated details. Do not add capabilities, notifications, "
        "fields, or behaviors that the user did not state. Acceptance criteria "
        "must be copied exactly from the user's answer. reply should briefly say "
        "the draft is ready. Use priority 2 unless the user explicitly supplied "
        "a priority. Leave area_path, iteration_path, and context null unless "
        "the user explicitly supplied them."
    )
    client = _client()
    result = None
    last_error = None
    content = ""
    for attempt in range(2):
        messages_for_model = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": conversation},
        ]
        if attempt:
            messages_for_model.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Correct your previous answer. Return valid JSON with "
                            "all required keys. For action ask, reply must contain "
                            "one clear question. For action draft, acceptance_criteria "
                            "must be a string or a list of strings."
                        ),
                    },
                ]
            )
        response = client.chat.completions.create(
            model=_required_env("OLLAMA_MODEL"),
            messages=messages_for_model,
        )
        try:
            content = response.choices[0].message.content or ""
        except IndexError as exc:
            last_error = RuntimeError("The model returned no choices.")
            continue
        try:
            candidate = _decode_json_response(content)
            if any(
                not isinstance(candidate.get(field), str)
                or not candidate[field].strip()
                for field in ("title", "description")
            ):
                raise RuntimeError(
                    "The model omitted the title or description from its draft."
                )
            candidate["action"] = "draft"
            result = candidate
            break
        except RuntimeError as exc:
            last_error = exc
    if result is None:
        raise RuntimeError(f"The model could not prepare a valid chat response: {last_error}")
    if not result.get("reply") and result.get("question"):
        result["reply"] = result["question"]
    if result.get("action") == "draft" and not result.get("reply"):
        result["reply"] = "Here is the complete user story draft for your review."
    if not isinstance(result.get("reply"), str) or not result["reply"].strip():
        raise RuntimeError("The model did not return a chat response.")
    if result["action"] == "draft":
        result["acceptance_criteria"] = acceptance_criteria
        user_text = "\n".join(_chat_user_text(messages))
        priority_match = re.search(
            r"\bpriority\s*[:=]?\s*([1-4])\b", user_text, re.IGNORECASE
        )
        result["priority"] = int(priority_match.group(1)) if priority_match else 2
        area_match = re.search(
            r"\barea\s+path\s*:\s*([^\r\n]+)", user_text, re.IGNORECASE
        )
        iteration_match = re.search(
            r"\biteration\s+path\s*:\s*([^\r\n]+)", user_text, re.IGNORECASE
        )
        context_match = re.search(
            r"\b(?:additional\s+)?context\s*:\s*([^\r\n]+)",
            user_text,
            re.IGNORECASE,
        )
        result["area_path"] = area_match.group(1).strip() if area_match else None
        result["iteration_path"] = (
            iteration_match.group(1).strip() if iteration_match else None
        )
        result["context"] = context_match.group(1).strip() if context_match else None
        required = ("title", "description", "acceptance_criteria")
        missing = [
            field
            for field in required
            if not isinstance(result.get(field), str) or not result[field].strip()
        ]
        if missing:
            raise RuntimeError(
                "The model returned an incomplete story draft: "
                + ", ".join(missing)
            )
        priority = result.get("priority", 2)
        if isinstance(priority, str):
            normalized = priority.strip().lower()
            priority = (
                1 if normalized in {"1", "critical", "urgent", "high"}
                else 2 if normalized in {"2", "medium", "normal"}
                else 3 if normalized in {"3", "low"}
                else 4 if normalized in {"4", "lowest"}
                else None
            )
        if not isinstance(priority, int) or priority not in {1, 2, 3, 4}:
            raise RuntimeError("The model returned an invalid Azure Boards priority.")
        result["priority"] = priority
    return result


def _route(state: AgentState) -> str:
    if state["plan"]["action"] == "clarify":
        return "clarify"
    return "execute"


def _clarify(state: AgentState) -> AgentState:
    return {"result": state["plan"].get("clarification") or "Please clarify your request."}


def _fetch_story(state: AgentState) -> AgentState:
    work_item_id = state.get("work_item_id")
    if not isinstance(work_item_id, int) or work_item_id < 1:
        return {"result": "Rejected: enter a valid positive Azure Boards work item ID."}

    item = get_user_story(work_item_id)
    fields = item.get("fields", {})
    item_type = fields.get("System.WorkItemType", "")
    expected_type = os.getenv("AZURE_DEVOPS_WORK_ITEM_TYPE", "User Story")
    if item_type.casefold() != expected_type.casefold():
        return {
            "result": (
                f"Rejected work item #{work_item_id}: expected a {expected_type}, "
                f"but Azure Boards reports {item_type or 'an unknown work item type'}."
            )
        }

    state_name = fields.get("System.State", "")
    reason = fields.get("System.Reason", "")
    if state_name.casefold() != "approved":
        explanation = f" Current reason: {reason}." if reason else ""
        return {
            "result": (
                f"Rejected user story #{work_item_id}: its Azure Boards state is "
                f"'{state_name or 'unspecified'}'; required state is 'Approved'."
                f"{explanation} No implementation was authorized."
            )
        }
    return {"story": item}


def _review_story(state: AgentState) -> AgentState:
    item = state["story"]
    fields = item["fields"]
    summary = (
        f"Approved Azure Boards user story #{item['id']}\n"
        f"Title: {fields.get('System.Title', '(no title)')}\n"
        f"Description: {fields.get('System.Description', '(no description)')}\n"
        f"Acceptance criteria: "
        f"{fields.get('Microsoft.VSTS.Common.AcceptanceCriteria', '(not provided)')}"
    )
    decision = interrupt(
        {
            "preview": summary,
            "prompt": "Approve handing this story to implementation? (yes/no)",
        }
    )
    if isinstance(decision, dict):
        decision = decision.get("action", "")
    if str(decision).strip().lower() not in {"approve", "yes", "y"}:
        return {
            "result": (
                f"Implementation rejected for user story #{item['id']} by human review. "
                "Azure Boards was not changed."
            )
        }
    return {
        "result": (
            f"Implementation approved for user story #{item['id']}. "
            "This agent has no code-editing or repository integration configured, "
            "so no source files were changed."
        )
    }


def _entry_route(state: AgentState) -> str:
    if state.get("plan"):
        return "execute"
    return "fetch_story" if state.get("mode") == "implement" else "parse"


def _execute(state: AgentState) -> AgentState:
    plan = state["plan"]
    action = plan["action"]
    validate_configuration()
    editable_fields = (
        "title",
        "description",
        "acceptance_criteria",
        "priority",
        "area_path",
        "iteration_path",
    )
    while True:
        if action == "create":
            preview = (
                f"Create Azure Boards user story\nTitle: {plan['title']}\n"
                f"Description: {plan['description']}\n"
                f"Acceptance criteria: {plan.get('acceptance_criteria') or '(not provided)'}\n"
                f"Priority: {plan.get('priority') or '(default)'}\n"
                f"Area path: {plan.get('area_path') or '(project default)'}\n"
                f"Iteration path: {plan.get('iteration_path') or '(project default)'}"
            )
        else:
            preview = (
                f"Update Azure Boards work item #{plan['work_item_id']}\n"
                f"Changes: {', '.join(key for key in editable_fields if plan.get(key) is not None)}"
            )
            for key in editable_fields:
                if plan.get(key) is not None:
                    preview += f"\n{key.replace('_', ' ').title()}: {plan[key]}"

        decision = interrupt(
            {
                "preview": preview,
                "prompt": "Choose approve, edit, or reject.",
                "editable_fields": editable_fields,
            }
        )
        if isinstance(decision, str):
            decision = {"action": decision.strip().lower()}
        if not isinstance(decision, dict):
            raise ValueError("Human review must return an approve, edit, or reject decision.")

        choice = decision.get("action", "").strip().lower()
        edits = decision.get("fields", {})
        if not isinstance(edits, dict):
            raise ValueError("Human edits must be provided as a fields object.")
        unknown_fields = set(edits) - set(editable_fields)
        if unknown_fields:
            raise ValueError(f"Unsupported human-edited fields: {', '.join(sorted(unknown_fields))}")
        for field, value in edits.items():
            if field == "priority":
                if not isinstance(value, int) or value not in {1, 2, 3, 4}:
                    raise ValueError("Priority must be an integer from 1 to 4.")
                plan[field] = value
            elif value is None and field in {"area_path", "iteration_path"}:
                plan[field] = None
            elif isinstance(value, str):
                plan[field] = value.strip() or None
            else:
                raise ValueError(f"Edited {field} must be text.")

        if choice in {"approve", "yes", "y"}:
            if action == "create" and (
                not plan.get("title")
                or not plan.get("description")
                or not plan.get("acceptance_criteria")
            ):
                raise ValueError(
                    "A story requires a title, description, and acceptance criteria."
                )
            break
        if choice in {"reject", "no", "n"}:
            return {"result": "Cancelled; Azure Boards was not changed."}
        if choice != "edit":
            raise ValueError("Human review decision must be approve, edit, or reject.")
        if action == "create" and (
            not plan.get("title")
            or not plan.get("description")
            or not plan.get("acceptance_criteria")
        ):
            raise ValueError(
                "A story requires a title, description, and acceptance criteria."
            )

    item = create_user_story(plan) if action == "create" else update_user_story(plan)
    identifier = item.get("id", plan.get("work_item_id"))
    url = item.get("url", "")
    return {"result": f"Saved Azure Boards work item #{identifier}. {url}".strip()}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("fetch_story", _fetch_story)
    builder.add_node("review_story", _review_story)
    builder.add_node("parse", _parse_request)
    builder.add_node("clarify", _clarify)
    builder.add_node("execute", _execute)
    builder.add_conditional_edges(
        START,
        _entry_route,
        {
            "fetch_story": "fetch_story",
            "parse": "parse",
            "execute": "execute",
        },
    )
    builder.add_conditional_edges(
        "fetch_story",
        lambda state: "review" if state.get("story") else "done",
        {"review": "review_story", "done": END},
    )
    builder.add_edge("review_story", END)
    builder.add_conditional_edges(
        "parse", _route, {"clarify": "clarify", "execute": "execute"}
    )
    builder.add_edge("clarify", END)
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=MemorySaver())


def run_request(graph, request: str, thread_id: str | None = None) -> dict:
    return graph.invoke(
        {"request": request},
        config={"configurable": {"thread_id": thread_id or str(uuid4())}},
    )
