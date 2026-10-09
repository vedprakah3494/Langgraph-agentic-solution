import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langgraph.types import Command

from agent.azure_boards import AzureBoardsError
from agent.graph import (
    _decode_json_response,
    build_graph,
    prepare_chat_story,
)


class _Message:
    content = (
        '{"action":"create","work_item_id":null,"title":"Search products",'
        '"description":"As a shopper, I want to search products, so that I can find items.",'
        '"acceptance_criteria":"Matching products are listed.","clarification":null}'
    )


class _Completion:
    choices = [type("Choice", (), {"message": _Message()})()]


class _Chat:
    class completions:
        @staticmethod
        def create(**_kwargs):
            return _Completion()


class _Client:
    chat = _Chat()


class _FallbackChat:
    class completions:
        calls = 0

        @classmethod
        def create(cls, **_kwargs):
            cls.calls += 1
            if cls.calls == 1:
                content = (
                    '{"action":"clarify","work_item_id":null,"title":null,'
                    '"description":null,"acceptance_criteria":null,'
                    '"clarification":"Please provide the Azure Boards work item ID."}'
                )
            else:
                content = (
                    '{"action":"create","work_item_id":null,'
                    '"title":"Apply for leave","description":"As an employee, '
                    'I want to apply for leave and have my manager approve it, '
                    'so that my leave is authorized.","acceptance_criteria":null,'
                    '"clarification":null}'
                )
            return type(
                "Completion",
                (),
                {"choices": [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]},
            )()


class _FallbackClient:
    chat = _FallbackChat()


class GraphTests(unittest.TestCase):
    @patch.dict("os.environ", {"OLLAMA_MODEL": "gemma3:latest"})
    @patch("agent.graph._client")
    def test_chat_story_response_requires_complete_user_input(self, _client):
        _client.return_value.chat.completions.create.return_value = type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {
                                    "content": (
                                        '{"action":"draft","reply":"Here is the draft.",'
                                        '"title":"Leave request","description":"As an employee, '
                                        'I want to apply for leave, so that my manager can '
                                        'approve it.","acceptance_criteria":"The employee can '
                                        'submit a request.","priority":2,"area_path":null,'
                                        '"iteration_path":null,"context":null}'
                                    )
                                },
                            )()
                        },
                    )()
                ]
            },
        )()
        result = prepare_chat_story(
            [
                {
                    "role": "user",
                    "content": (
                        "Employee wants to apply for leave so that their leave is "
                        "authorized and tracked. Acceptance criteria: Employee "
                        "can submit leave dates and manager can approve or reject."
                    ),
                }
            ]
        )

        self.assertEqual(result["action"], "draft")
        self.assertTrue(result["title"])
        self.assertTrue(result["description"])
        self.assertEqual(
            result["acceptance_criteria"],
            "Employee can submit leave dates and manager can approve or reject.",
        )

    def test_chat_requests_acceptance_criteria_when_user_did_not_provide_them(self):
        result = prepare_chat_story(
            [
                {
                    "role": "user",
                    "content": "Employee wants to apply for leave so that it is tracked.",
                }
            ]
        )

        self.assertEqual(result["action"], "ask")
        self.assertIn("acceptance criterion", result["reply"])

    def test_chat_requests_benefit_when_user_did_not_provide_one(self):
        result = prepare_chat_story(
            [
                {
                    "role": "user",
                    "content": (
                        "Employee wants to apply for leave. Acceptance criteria: "
                        "Manager can approve or reject the request."
                    ),
                }
            ]
        )

        self.assertEqual(result["action"], "ask")
        self.assertIn("business benefit", result["reply"])

    def test_dotenv_path_is_independent_of_process_working_directory(self):
        from agent.graph import _DOTENV_PATH

        self.assertEqual(_DOTENV_PATH, Path(__file__).resolve().parents[1] / ".env")

    def test_decode_json_response_accepts_gemma_markdown_fence(self):
        self.assertEqual(
            _decode_json_response('```json\n{"action": "clarify"}\n```'),
            {"action": "clarify"},
        )

    @patch.dict(
        "os.environ",
        {
            "OLLAMA_MODEL": "gemma3",
            "AZURE_DEVOPS_ORGANIZATION": "https://dev.azure.com/test",
            "AZURE_DEVOPS_PROJECT": "Test",
            "AZURE_DEVOPS_PAT": "test-pat",
        },
    )
    @patch("agent.graph._client", return_value=_FallbackClient())
    def test_story_description_retries_as_create_not_update(self, _client):
        _FallbackChat.completions.calls = 0
        from agent.graph import _parse_request

        result = _parse_request(
            {"request": "employe want apply and manager approve"}
        )

        self.assertEqual(result["plan"]["action"], "create")
        self.assertEqual(result["plan"]["title"], "Apply for leave")
        self.assertEqual(_FallbackChat.completions.calls, 2)

    @patch("agent.graph.create_user_story")
    @patch.dict(
        "os.environ",
        {
            "OLLAMA_MODEL": "gemma3",
            "AZURE_DEVOPS_ORGANIZATION": "",
            "AZURE_DEVOPS_PROJECT": "",
            "AZURE_DEVOPS_PAT": "",
        },
    )
    @patch("agent.graph._client", return_value=_Client())
    def test_missing_azure_settings_fail_before_human_approval(
        self, _client, create_story
    ):
        graph = build_graph()
        config = {"configurable": {"thread_id": "missing-azure-config-test"}}

        with self.assertRaisesRegex(AzureBoardsError, "AZURE_DEVOPS_ORGANIZATION"):
            graph.invoke({"request": "Create a product search story"}, config=config)

        create_story.assert_not_called()

    @patch("agent.graph.create_user_story", return_value={"id": 42, "url": "https://boards/42"})
    @patch.dict(
        "os.environ",
        {
            "OLLAMA_MODEL": "gemma3",
            "AZURE_DEVOPS_ORGANIZATION": "https://dev.azure.com/test",
            "AZURE_DEVOPS_PROJECT": "Test",
            "AZURE_DEVOPS_PAT": "test-pat",
        },
    )
    @patch("agent.graph._client", return_value=_Client())
    def test_story_is_not_written_until_approved(self, _client, create_story):
        graph = build_graph()
        config = {"configurable": {"thread_id": "approval-test"}}
        state = graph.invoke({"request": "Create a product search story"}, config=config)

        self.assertIn("__interrupt__", state)
        create_story.assert_not_called()

        state = graph.invoke(Command(resume={"action": "approve"}), config=config)
        create_story.assert_called_once()
        self.assertIn("Saved Azure Boards work item #42", state["result"])

    @patch(
        "agent.graph.create_user_story",
        return_value={"id": 82, "url": "https://boards/82"},
    )
    @patch.dict(
        "os.environ",
        {
            "OLLAMA_MODEL": "gemma3",
            "AZURE_DEVOPS_ORGANIZATION": "https://dev.azure.com/test",
            "AZURE_DEVOPS_PROJECT": "Test",
            "AZURE_DEVOPS_PAT": "test-pat",
        },
    )
    def test_chat_draft_is_created_only_after_approval(self, create_story):
        graph = build_graph()
        config = {"configurable": {"thread_id": "chat-approval-test"}}
        plan = {
            "action": "create",
            "title": "Leave request",
            "description": "As an employee, I want to request leave.",
            "acceptance_criteria": "A manager can approve or reject the request.",
            "priority": 2,
            "area_path": None,
            "iteration_path": None,
            "context": "Requests include start and end dates.",
        }

        state = graph.invoke({"plan": plan}, config=config)

        self.assertIn("__interrupt__", state)
        create_story.assert_not_called()

        state = graph.invoke(
            Command(
                resume={
                    "action": "approve",
                    "fields": {
                        field: plan.get(field)
                        for field in (
                            "title",
                            "description",
                            "acceptance_criteria",
                            "priority",
                            "area_path",
                            "iteration_path",
                        )
                    },
                }
            ),
            config=config,
        )

        create_story.assert_called_once()
        self.assertIn("#82", state["result"])

    @patch(
        "agent.graph.get_user_story",
        return_value={
            "id": 73,
            "url": "https://boards/73",
            "fields": {
                "System.WorkItemType": "User Story",
                "System.State": "Active",
                "System.Reason": "Work started",
            },
        },
    )
    def test_unapproved_story_is_rejected_with_reason(self, get_story):
        graph = build_graph()
        state = graph.invoke(
            {"mode": "implement", "work_item_id": 73},
            config={"configurable": {"thread_id": "unapproved-story-test"}},
        )

        get_story.assert_called_once_with(73)
        self.assertIn("state is 'Active'", state["result"])
        self.assertIn("Current reason: Work started", state["result"])
        self.assertIn("No implementation was authorized", state["result"])

    @patch(
        "agent.graph.get_user_story",
        return_value={
            "id": 73,
            "url": "https://boards/73",
            "fields": {
                "System.WorkItemType": "User Story",
                "System.State": "Approved",
                "System.Title": "Search products",
                "System.Description": "As a shopper, I want to search.",
                "Microsoft.VSTS.Common.AcceptanceCriteria": "Matching items appear.",
            },
        },
    )
    def test_approved_story_requires_human_handoff_approval(self, get_story):
        graph = build_graph()
        config = {"configurable": {"thread_id": "approved-story-test"}}
        state = graph.invoke(
            {"mode": "implement", "work_item_id": 73}, config=config
        )

        get_story.assert_called_once_with(73)
        self.assertIn("__interrupt__", state)
        self.assertIn("Search products", state["__interrupt__"][0].value["preview"])

        state = graph.invoke(Command(resume="yes"), config=config)

        self.assertIn("Implementation approved for user story #73", state["result"])
        self.assertIn("no code-editing", state["result"])

    @patch("agent.graph.create_user_story")
    @patch.dict(
        "os.environ",
        {
            "OLLAMA_MODEL": "gemma3",
            "AZURE_DEVOPS_ORGANIZATION": "https://dev.azure.com/test",
            "AZURE_DEVOPS_PROJECT": "Test",
            "AZURE_DEVOPS_PAT": "test-pat",
        },
    )
    @patch("agent.graph._client", return_value=_Client())
    def test_declining_preview_does_not_write(self, _client, create_story):
        graph = build_graph()
        config = {"configurable": {"thread_id": "decline-test"}}
        graph.invoke({"request": "Create a product search story"}, config=config)

        state = graph.invoke(Command(resume={"action": "reject"}), config=config)

        create_story.assert_not_called()
        self.assertEqual(state["result"], "Cancelled; Azure Boards was not changed.")

    @patch("agent.graph.create_user_story", return_value={"id": 42})
    @patch.dict(
        "os.environ",
        {
            "OLLAMA_MODEL": "gemma3",
            "AZURE_DEVOPS_ORGANIZATION": "https://dev.azure.com/test",
            "AZURE_DEVOPS_PROJECT": "Test",
            "AZURE_DEVOPS_PAT": "test-pat",
        },
    )
    @patch("agent.graph._client", return_value=_Client())
    def test_human_can_edit_story_before_approval(self, _client, create_story):
        graph = build_graph()
        config = {"configurable": {"thread_id": "edit-test"}}
        state = graph.invoke({"request": "Create a product search story"}, config=config)

        state = graph.invoke(
            Command(
                resume={
                    "action": "edit",
                    "fields": {
                        "title": "Find products",
                        "acceptance_criteria": "Only available products appear.",
                    },
                }
            ),
            config=config,
        )

        self.assertIn("__interrupt__", state)
        self.assertIn("Title: Find products", state["__interrupt__"][0].value["preview"])
        create_story.assert_not_called()

        state = graph.invoke(Command(resume={"action": "approve"}), config=config)
        create_story.assert_called_once()
        self.assertEqual(create_story.call_args.args[0]["title"], "Find products")
        self.assertEqual(
            create_story.call_args.args[0]["acceptance_criteria"],
            "Only available products appear.",
        )
        self.assertIn("Saved Azure Boards work item #42", state["result"])


if __name__ == "__main__":
    unittest.main()
