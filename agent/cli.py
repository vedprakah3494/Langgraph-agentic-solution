import sys
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.azure_boards import AzureBoardsError
from agent.graph import build_graph, prepare_chat_story


def _story_creation_chat() -> dict | None:
    messages = []
    print(
        "\nDescribe the user story in your own words. "
        "Type 'cancel' at any time to stop."
    )
    prompt = input("\nYou: ").strip()
    if not prompt or prompt.lower() == "cancel":
        return None
    messages.append({"role": "user", "content": prompt})

    while True:
        result = prepare_chat_story(messages)
        if result["action"] == "draft":
            print(f"\nAssistant: {result['reply']}\n")
            print(f"Title: {result['title']}")
            print(f"User story: {result['description']}")
            print(f"Acceptance criteria:\n{result['acceptance_criteria']}")
            print(f"Priority: {result['priority']}")
            print(f"Area path: {result.get('area_path') or 'Project default'}")
            print(f"Iteration path: {result.get('iteration_path') or 'Project default'}")
            if result.get("context"):
                print(f"Additional context: {result['context']}")
            return {
                "action": "create",
                "title": result["title"],
                "description": result["description"],
                "acceptance_criteria": result["acceptance_criteria"],
                "priority": result["priority"],
                "area_path": result.get("area_path"),
                "iteration_path": result.get("iteration_path"),
            }

        reply = result["reply"]
        print(f"\nAssistant: {reply}")
        answer = input("\nYou: ").strip()
        if not answer or answer.lower() == "cancel":
            return None
        messages.extend(
            [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": answer},
            ]
        )


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid4())}}
    mode = input(
        "Choose a workflow: [1] create a user story, [2] update a story, "
        "[3] check approval of an existing story: "
    ).strip()
    if mode == "3":
        story_id = input("Azure Boards user story ID: ").strip()
        if not story_id.isdecimal() or int(story_id) < 1:
            raise SystemExit("Enter a valid positive Azure Boards work item ID.")
        initial_state = {"mode": "implement", "work_item_id": int(story_id)}
    elif mode == "1":
        plan = _story_creation_chat()
        if plan is None:
            print("Story creation cancelled.")
            return
        initial_state = {"plan": plan}
    elif mode == "2":
        request = input("Describe the change and include the work item ID: ").strip()
        if not request:
            raise SystemExit("A request is required.")
        initial_state = {"request": request}
    else:
        raise SystemExit("Choose workflow 1, 2, or 3.")
    try:
        state = graph.invoke(initial_state, config=config)
        while state.get("__interrupt__"):
            details = state["__interrupt__"][0].value
            print(f"\n{details['preview']}\n")
            choice = input(f"{details['prompt']} ").strip().lower()
            if choice in {"approve", "yes", "y"}:
                response = {"action": "approve"}
            elif choice in {"reject", "no", "n"}:
                response = {"action": "reject"}
            elif choice == "edit" and "editable_fields" not in details:
                print("Enter yes to approve or no to reject.")
                continue
            elif choice == "edit":
                fields = {}
                for field in details["editable_fields"]:
                    current = input(
                        f"{field.replace('_', ' ').title()} "
                        f"[{state['plan'].get(field) or '(not set)'}]: "
                    )
                    if current.strip():
                        fields[field] = current.strip()
                response = {"action": "edit", "fields": fields}
            else:
                print("Enter approve, edit, or reject.")
                continue
            state = graph.invoke(Command(resume=response), config=config)
    except AzureBoardsError as exc:
        raise SystemExit(f"Azure Boards configuration/request error: {exc}") from exc
    print(state["result"])


if __name__ == "__main__":
    main()
