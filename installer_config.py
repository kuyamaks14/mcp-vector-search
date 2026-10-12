"""Small, dependency-free renderers for Claude Code and Codex MCP config."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def build_server_entry(command: str, server_path: str, home: str) -> dict[str, Any]:
    return {
        "command": command,
        "args": [server_path],
        "env": {"GO_CONFLUENCE_MCP_HOME": home, "HF_HOME": f"{home}/models"},
    }


def merge_claude_config(existing: str, name: str, entry: Mapping[str, Any]) -> str:
    data = json.loads(existing) if existing.strip() else {}
    if not isinstance(data, dict):
        raise ValueError("Claude config root must be an object")
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError("mcpServers must be an object")
    servers[name] = {"type": "stdio", **entry}
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def merge_codex_config(existing: str, name: str, entry: Mapping[str, Any]) -> str:
    begin = "# BEGIN confluence-rag-mcp"
    end = "# END confluence-rag-mcp"
    lines = [
        begin,
        f"[mcp_servers.{name}]",
        f"command = {_toml_string(str(entry['command']))}",
        "args = [" + ", ".join(_toml_string(item) for item in entry["args"]) + "]",
        "env = {" + ", ".join(
            f"{key} = {_toml_string(value)}" for key, value in entry["env"].items()
        ) + "}",
        end,
    ]
    block = "\n".join(lines) + "\n"
    start = existing.find(begin)
    if start < 0:
        return existing.rstrip() + "\n\n" + block
    finish = existing.find(end, start)
    if finish < 0:
        raise ValueError("managed Codex block is incomplete")
    finish += len(end)
    return existing[:start] + block + existing[finish:].lstrip("\n")
