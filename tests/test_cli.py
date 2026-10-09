import unittest
from unittest.mock import call, patch

from agent.cli import _story_creation_chat


class CliTests(unittest.TestCase):
    @patch(
        "agent.cli.prepare_chat_story",
        side_effect=[
            {
                "action": "ask",
                "reply": "What acceptance criteria should be met?",
            },
            {
                "action": "draft",
                "reply": "Draft ready.",
                "title": "Leave request",
                "description": "As an employee, I want to request leave.",
                "acceptance_criteria": "A manager can approve or reject.",
                "priority": 2,
                "area_path": None,
                "iteration_path": None,
                "context": None,
            },
        ],
    )
    @patch("builtins.input", side_effect=["Employee needs leave.", "Manager can approve."])
    def test_story_chat_asks_and_returns_draft_for_approval(self, _input, prepare):
        plan = _story_creation_chat()

        self.assertEqual(plan["action"], "create")
        self.assertEqual(plan["title"], "Leave request")
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(
            prepare.call_args_list[1].args[0][-1],
            {"role": "user", "content": "Manager can approve."},
        )

    @patch("builtins.input", return_value="cancel")
    def test_story_chat_can_be_cancelled(self, _input):
        self.assertIsNone(_story_creation_chat())


if __name__ == "__main__":
    unittest.main()
