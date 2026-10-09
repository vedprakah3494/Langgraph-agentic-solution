# Azure Boards User Story Agent

A LangGraph command-line agent that turns a natural-language request into a
proposed Azure Boards user story or update. It asks for clarification when the
target or essential details are missing and always previews the change for
approval before writing to Azure Boards.

## Setup

1. Use Python 3.10 or newer and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`. Set `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, and
   `OLLAMA_MODEL` in `.env` for your Ollama server and selected model. Install
   that model in Ollama, then ensure the server is running.

3. Set the Azure DevOps organization URL, project, and PAT in `.env`.

   The PAT needs **Work Items: Read & write** access. Keep `.env` private and
   never commit it. These Azure Boards settings are required before the agent
   can show the approval preview or save a story.

4. Start the command-line chat from the project root:

   ```powershell
   python -m agent.cli
   ```

You can also run `python agent\cli.py` from the project root.

## Workflows

### User story chat

Choose workflow **1** and describe the requested feature naturally. The
assistant asks follow-up questions for missing persona, need, business value,
or testable acceptance criteria instead of filling gaps with guesses. Continue
in the terminal chat until it has enough detail. Type `cancel` to stop.

The agent then replies with the complete story: title, “As a / I want / so
that” description, acceptance criteria, priority, and any area/iteration paths
or context. Review those details in the terminal, then approve or reject when
prompted. The Azure Boards work item is created only after approval. This
approval creates a Boards item; it is not a Git commit.

When you run `python -m agent.cli`, choose one of these workflows:

### 1. Create a user story

Use the multi-turn story chat described above. At the final review prompt:

- Enter **approve** to save the changes to Azure Boards.
- Enter **reject** to cancel without changing Azure Boards.

New work items use the configured `AZURE_DEVOPS_WORK_ITEM_TYPE` (defaults to
`User Story`).

### 2. Update an existing story

Describe the changes and include the Azure Boards work item ID. Review the
proposed fields, then approve or reject the update.

### 3. Check approval for an existing story

Enter the Azure Boards work item ID. The agent fetches the work item and checks
that its type matches `AZURE_DEVOPS_WORK_ITEM_TYPE` and its `System.State` is
`Approved` (case-insensitive).

- If the type or state does not match, the agent rejects the request, reports
  the current state and `System.Reason` when available, and does not proceed.
- If the story is approved, the agent displays its title, description, and
  acceptance criteria. Enter **yes** to approve the handoff or **no** to reject
  it.

This approval workflow only validates the Azure Boards state and records your
handoff decision. It does **not** implement the story, modify source code, or
create a Git commit; repository integration is not configured in this agent.

## Configuration

| Variable | Purpose |
| --- | --- |
| `OLLAMA_BASE_URL` | Ollama OpenAI-compatible endpoint, read from `.env` |
| `OLLAMA_API_KEY` | API key for Ollama-compatible servers, read from `.env` |
| `OLLAMA_MODEL` | Ollama model name, read from `.env` |
| `AZURE_DEVOPS_ORGANIZATION` | Organization URL, e.g. `https://dev.azure.com/contoso` |
| `AZURE_DEVOPS_PROJECT` | Azure DevOps project name |
| `AZURE_DEVOPS_PAT` | PAT with Work Items read/write permission |
| `AZURE_DEVOPS_WORK_ITEM_TYPE` | Process work-item type (defaults to `User Story`) |
