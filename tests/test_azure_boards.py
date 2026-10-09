import unittest
from unittest.mock import patch

from agent.azure_boards import _format_html, _patch_fields, get_user_story


class AzureBoardsTests(unittest.TestCase):
    def test_description_is_html_escaped(self):
        self.assertEqual(_format_html("<script>\nhello"), "<p>&lt;script&gt;</p><p>hello</p>")

    def test_create_patch_contains_only_provided_fields(self):
        operations = _patch_fields(
            {"title": "Search", "description": "As a user, I want search."}
        )
        self.assertEqual(len(operations), 2)
        self.assertEqual(operations[0]["op"], "add")
        self.assertEqual(operations[1]["value"], "<p>As a user, I want search.</p>")

    def test_update_patch_does_not_overwrite_unspecified_fields(self):
        operations = _patch_fields({"acceptance_criteria": "It works"})
        self.assertEqual(
            operations,
            [
                {
                    "op": "add",
                    "path": "/fields/Microsoft.VSTS.Common.AcceptanceCriteria",
                    "value": "<p>It works</p>",
                }
            ],
        )

    def test_planning_fields_keep_their_azure_boards_types(self):
        operations = _patch_fields(
            {
                "title": "Apply for leave",
                "description": "As an employee, I want to apply.",
                "acceptance_criteria": "Manager can approve.",
                "priority": 2,
                "area_path": "Project\\HR",
                "iteration_path": "Project\\Sprint 1",
            }
        )

        fields = {item["path"]: item["value"] for item in operations}
        self.assertEqual(fields["/fields/Microsoft.VSTS.Common.Priority"], 2)
        self.assertEqual(fields["/fields/System.AreaPath"], "Project\\HR")
        self.assertEqual(fields["/fields/System.IterationPath"], "Project\\Sprint 1")
        self.assertEqual(
            fields["/fields/Microsoft.VSTS.Common.AcceptanceCriteria"],
            "<p>Manager can approve.</p>",
        )

    @patch.dict(
        "os.environ",
        {
            "AZURE_DEVOPS_ORGANIZATION": "https://dev.azure.com/example",
            "AZURE_DEVOPS_PROJECT": "Sample Project",
            "AZURE_DEVOPS_PAT": "test-token",
        },
    )
    @patch("agent.azure_boards._request", return_value={"id": 73})
    def test_fetch_story_requests_fields_from_azure_boards(self, request):
        result = get_user_story(73)

        self.assertEqual(result, {"id": 73})
        request.assert_called_once()
        method, url, _pat = request.call_args.args
        self.assertEqual(method, "GET")
        self.assertIn("/Sample%20Project/_apis/wit/workitems/73?", url)
        self.assertIn("$expand=fields", url)


if __name__ == "__main__":
    unittest.main()
