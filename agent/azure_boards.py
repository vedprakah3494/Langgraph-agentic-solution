import base64
import json
import os
from html import escape
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class AzureBoardsError(RuntimeError):
    pass


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise AzureBoardsError(f"Missing required environment variable: {name}")
    return value


def validate_configuration() -> None:
    for name in (
        "AZURE_DEVOPS_ORGANIZATION",
        "AZURE_DEVOPS_PROJECT",
        "AZURE_DEVOPS_PAT",
    ):
        _required_env(name)


def _format_html(value: str) -> str:
    return "<p>" + "</p><p>".join(escape(part) for part in value.splitlines()) + "</p>"


def _request(method: str, url: str, pat: str, body: list | None = None) -> dict:
    token = base64.b64encode(f":{pat}".encode("utf-8")).decode("ascii")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json-patch+json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AzureBoardsError(
            f"Azure Boards returned HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise AzureBoardsError(f"Could not reach Azure Boards: {exc.reason}") from exc


def _base_url() -> str:
    organization = _required_env("AZURE_DEVOPS_ORGANIZATION").rstrip("/")
    project = quote(_required_env("AZURE_DEVOPS_PROJECT"), safe="")
    return f"{organization}/{project}/_apis/wit"


def _patch_fields(plan: dict) -> list[dict]:
    fields = []
    for key, value in (
        ("System.Title", plan.get("title")),
        ("System.Description", plan.get("description")),
        ("Microsoft.VSTS.Common.AcceptanceCriteria", plan.get("acceptance_criteria")),
        ("Microsoft.VSTS.Common.Priority", plan.get("priority")),
        ("System.AreaPath", plan.get("area_path")),
        ("System.IterationPath", plan.get("iteration_path")),
    ):
        if value is not None:
            if key in {
                "System.Description",
                "Microsoft.VSTS.Common.AcceptanceCriteria",
            }:
                value = _format_html(value)
            fields.append(
                {
                    "op": "add",
                    "path": f"/fields/{key}",
                    "value": value,
                }
            )
    return fields


def create_user_story(plan: dict) -> dict:
    pat = _required_env("AZURE_DEVOPS_PAT")
    work_item_type = quote(
        os.getenv("AZURE_DEVOPS_WORK_ITEM_TYPE", "User Story"), safe=""
    )
    url = (
        f"{_base_url()}/workitems/${work_item_type}"
        "?api-version=7.1"
    )
    return _request("POST", url, pat, _patch_fields(plan))


def update_user_story(plan: dict) -> dict:
    pat = _required_env("AZURE_DEVOPS_PAT")
    work_item_id = plan["work_item_id"]
    url = f"{_base_url()}/workitems/{work_item_id}?api-version=7.1"
    return _request("PATCH", url, pat, _patch_fields(plan))


def get_user_story(work_item_id: int) -> dict:
    if not isinstance(work_item_id, int) or work_item_id < 1:
        raise AzureBoardsError("Azure Boards work item ID must be a positive integer.")
    pat = _required_env("AZURE_DEVOPS_PAT")
    url = f"{_base_url()}/workitems/{work_item_id}?$expand=fields&api-version=7.1"
    return _request("GET", url, pat)
