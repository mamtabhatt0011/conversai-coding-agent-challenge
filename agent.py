"""
TOOLS
The agent exposes a small JSON action protocol over the supplied repository tools.
It preloads the repository map, then lets the model inspect, edit, and test through
those interfaces only.

LOOP
Each model turn returns one action. Reads provide focused context; edits are followed
by tests automatically so the next turn sees concrete validation. The loop keeps a
short, explicit history and reserves the final turns for repair.

FAILURE HANDLING
Malformed JSON, invalid actions, failed patches, and test failures become tool-result
feedback. Edits are never guessed or applied by this wrapper. The model must recover
from exact tool errors and test output. A successful test run is strong completion
evidence, while a premature finish is accepted only after the model has had a chance
to validate an edit.

STOPPING
Stop on passing tests after an edit, or on an explicit finish after useful validation.
Otherwise stop at five model calls or the tool budget. The wrapper never accesses the
filesystem, subprocesses, network, or an alternate model directly.
"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """
You are a careful coding agent working inside a Python repository.

Goal: solve the user's issue completely, with the smallest general source-code change.
You have exactly five model turns. Inspect implementation and relevant visible tests
before editing. Never edit tests. Anticipate hidden regression tests.

Return exactly ONE JSON object and no markdown:
{"action":"list_files"}
{"action":"read_file","path":"relative/path.py"}
{"action":"read_many","paths":["a.py","b.py"]}
{"action":"write_file","path":"relative/path.py","content":"complete file"}
{"action":"patch_file","path":"relative/path.py","replacements":[{"old":"exact text","new":"replacement"}]}
{"action":"run_tests"}
{"action":"finish","summary":"brief result"}

Rules:
- The repository file list is already supplied in the conversation; do not list it
  again unless genuinely needed.
- Prefer read_many when several small relevant files are needed.
- Read the visible test and implementation before changing behavior.
- Prefer patch_file for focused edits; use write_file only when replacing a small file.
- Every replacement old string must match exactly. Do not invent unseen source.
- After editing, validation is automatic; use the resulting test output to repair.
- Do not claim success without tests passing, unless no meaningful edit was needed.
- Do not hardcode a public task name or visible assertion.
- Paths must be relative repository paths.
""".strip()


def parse_action(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    # Be tolerant of a short prose prefix/suffix while still requiring one object.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Expected one JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict) or not isinstance(value.get("action"), str):
        raise ValueError("Expected a JSON object containing an action")
    return value


def _read_many(paths: Any, tools: Any) -> str:
    if not isinstance(paths, list) or not paths:
        raise ValueError("read_many requires a non-empty paths list")
    if len(paths) > 4:
        raise ValueError("read_many accepts at most four paths")
    chunks = []
    for path in paths:
        if not isinstance(path, str) or not path:
            raise ValueError("read_many paths must be non-empty strings")
        chunks.append(f"===== {path} =====\n{tools.read_file(path)}")
    return "\n\n".join(chunks)


def execute_action(action: dict[str, Any], tools: Any) -> tuple[str, bool]:
    name = action["action"]

    if name == "list_files":
        return tools.list_files(), False

    if name == "read_file":
        path = str(action.get("path", ""))
        if not path:
            raise ValueError("read_file requires path")
        return tools.read_file(path), False

    if name == "read_many":
        return _read_many(action.get("paths"), tools), False

    if name == "write_file":
        path = str(action.get("path", ""))
        if not path:
            raise ValueError("write_file requires path")
        content = action.get("content")
        if not isinstance(content, str):
            raise ValueError("write_file requires string content")
        result = tools.write_file(path, content)
        return result, True

    if name == "patch_file":
        path = str(action.get("path", ""))
        replacements = action.get("replacements")
        if not path or not isinstance(replacements, list) or not replacements:
            raise ValueError("patch_file requires path and replacements")
        if len(replacements) > 8:
            raise ValueError("patch_file accepts at most eight replacements")
        for item in replacements:
            if not isinstance(item, dict):
                raise ValueError("each replacement must be an object")
            if not isinstance(item.get("old"), str) or not isinstance(item.get("new"), str):
                raise ValueError("replacement requires string old/new")
        result = tools.patch_file(path, replacements)
        return result, True

    if name == "run_tests":
        return tools.run_tests(), False

    if name == "finish":
        return str(action.get("summary", "Finished")), False

    raise ValueError(f"Unknown action: {name}")


def solve(task: str, tools: Any, llm: Any) -> str:
    # One deterministic discovery call gives the model a map without spending an
    # LLM turn asking it to discover the repository shape.
    try:
        repo_map = tools.list_files()
    except Exception as error:
        repo_map = f"REPO_MAP_ERROR: {type(error).__name__}: {error}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "USER TASK:\n" + task +
                "\n\nREPOSITORY FILE MAP:\n" + repo_map +
                "\n\nBegin by inspecting the most relevant implementation and visible test files."
            ),
        },
    ]

    for turn in range(5):
        try:
            response = llm.ask(messages)
        except Exception as error:
            return f"LLM_ERROR: {type(error).__name__}: {error}"

        messages.append({"role": "assistant", "content": response})
        try:
            action = parse_action(response)
            name = action["action"]

            if name == "finish":
                return str(action.get("summary", "Finished"))

            result, edited = execute_action(action, tools)

            # Validation is coupled to mutation: this prevents spending a model
            # turn merely deciding to run tests and leaves repair turns available.
            if edited:
                test_result = tools.run_tests()
                result = f"{result}\n\n===== AUTOMATIC TEST RUN =====\n{test_result}"
                if test_result.startswith("Exit code: 0"):
                    return "Implemented the fix and tests pass"

            messages.append({
                "role": "user",
                "content": (
                    f"Turn {turn + 1} tool result for `{name}`:\n{result}\n\n"
                    "Continue from this evidence. If tests fail, inspect/repair the "
                    "source and validate again. If solved, finish."
                ),
            })
        except Exception as error:
            messages.append({
                "role": "user",
                "content": (
                    f"ACTION_ERROR: {type(error).__name__}: {error}\n"
                    "Correct the action and continue; do not assume the edit happened."
                ),
            })

    return "Model-call limit reached; work may be incomplete."
