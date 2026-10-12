# Confluence RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local MCP server a persistent, documentation-only Confluence RAG for Claude Code and Codex.

**Architecture:** The host agent fetches Confluence pages through its configured Confluence MCP and submits normalized page payloads to the local stdio RAG server. The server keeps page-id keyed Chroma/BM25 state and exposes index, search and explicit reset tools. The installer chooses one client, installs the local server and safely adds that client's MCP configuration.

**Tech Stack:** Python 3.10+, official Python MCP SDK 1.x, sentence-transformers, ChromaDB, rank-bm25, Bash, Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-12-confluence-rag-design.md`

## Global Constraints

- Read Confluence data only through the agent-hosted Confluence MCP; the local server and installer make no direct Confluence HTTP requests and store no credentials.
- Keep the existing dependency versions and Python 3.10 floor; do not add dependencies.
- Preserve stdio JSON-RPC integrity: server diagnostics go only to stderr.
- The server indexes only Confluence page payloads; it must not expose Go-code or directory indexing.
- Keep storage persistent and page-id keyed; unchanged `version` plus content digest must not generate new embeddings.
- `clear_index` is destructive and its tool description must restrict it to an explicit user request.
- Do not modify the user's real Claude/Codex configuration while testing; use temporary config paths.
- Do not create Git commits unless the user explicitly asks for one.

## Review Focus

- A page rename with the same page id must replace metadata without leaving old chunks; cover in Task 2.
- An empty but valid Confluence page must be remembered without inserting a vector; cover in Task 2.
- A malformed page payload must be rejected before the vector store is changed; cover in Task 3.
- Re-running a chosen-client installer must retain unrelated MCP entries and create one managed RAG entry; cover in Task 4.
- Codex TOML values containing spaces or quotes in paths must remain valid and idempotently replace only the managed block; cover in Task 4.

## File Structure

- Modify: `hybrid_knowledge_mcp.py` — Confluence document splitting, page manifest, retrieval results, MCP tools and isolated self-test.
- Create: `installer_config.py` — deterministic Claude JSON and Codex TOML snippet/merge helpers used by the installer and unit tests.
- Modify: `install.sh` — client selection, copying `installer_config.py`, config snippet rendering/merge and bootstrap guidance.
- Modify: `README.md` — Confluence-only scope, two client installation paths and agent-mediated bootstrap instructions.
- Modify: `claude_mcp_config.example.json` — current server name and environment variables.
- Create: `codex_mcp_config.example.toml` — equivalent Codex MCP example.
- Create: `tests/test_hybrid_knowledge_mcp.py` — storage/chunking/search/MCP-input regression tests using a deterministic fake embedder and temporary data directories.
- Create: `tests/test_installer_config.py` — Claude JSON and Codex TOML rendering/merge tests.

### Task 1: Establish test harness and page chunking

**Files:**
- Create: `tests/test_hybrid_knowledge_mcp.py`
- Modify: `hybrid_knowledge_mcp.py: constants and Markdown chunking helpers`

**Interfaces:**
- Produces: `split_confluence_markdown(content: str, title: str, ancestors: Sequence[str]) -> list[tuple[str, dict[str, Any]]]`.
- Produces: injectable `KnowledgeBase(..., embedder: Embedder | None = None)` so tests can use a deterministic in-memory-compatible fake encoder.

- [ ] **Step 1: Write the failing Markdown page chunking tests**

Add tests for headings, ancestor breadcrumbs, a title without headings and an empty body. Assert every non-empty chunk carries the supplied title and `ancestor_path`, and that empty content produces no chunks.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python3 -m unittest tests.test_hybrid_knowledge_mcp.ConfluenceChunkingTests -v`

Expected: FAIL because the page splitter and test injection interface do not exist.

- [ ] **Step 3: Implement `split_confluence_markdown` and the test-only embedder injection seam**

Remove Go/file-type-specific splitting from the active page path. Reuse the existing Markdown heading-aware splitting mechanics, prepend a Confluence title/breadcrumb header to each document, and leave the production `Embedder` as the default.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python3 -m unittest tests.test_hybrid_knowledge_mcp.ConfluenceChunkingTests -v`

Expected: PASS.

### Task 2: Make the persistent index Confluence-page keyed

**Files:**
- Modify: `hybrid_knowledge_mcp.py: manifest, indexing, stats and clear sections`
- Test: `tests/test_hybrid_knowledge_mcp.py`

**Interfaces:**
- Consumes: `split_confluence_markdown(content, title, ancestors)` from Task 1.
- Produces: `KnowledgeBase.index_confluence_page(page_id: str, url: str, title: str, content: str, version: str | int | None, ancestors: Sequence[str]) -> PageIndexReport`.
- Produces: `PageIndexReport(status: Literal["indexed", "updated", "unchanged", "empty"], chunks_added: int, chunks_deleted: int, total_chunks: int, elapsed: float)`.

- [ ] **Step 1: Write failing page-index lifecycle tests**

Use temporary directories and a fake embedder to test: first index stores URL/title/ancestors and chunks; same version and digest returns `unchanged`; changed content/version deletes old chunk ids and returns `updated`; same id with a renamed title updates metadata; an empty page is in the manifest with no chunks; clear then index/search works again.

- [ ] **Step 2: Run lifecycle tests to verify they fail**

Run: `python3 -m unittest tests.test_hybrid_knowledge_mcp.PageIndexLifecycleTests -v`

Expected: FAIL because page-oriented indexing and reports do not exist.

- [ ] **Step 3: Replace directory/file ownership with page ownership**

Change the manifest schema from `files` keyed by paths to `pages` keyed by `page_id`; derive stable chunk ids only from the page id. On update delete only that page's recorded chunks, add the new chunks, rebuild/persist BM25, and persist metadata including digest/version/title/url/ancestors. Update `stats()` and `clear()` terminology and state accordingly.

- [ ] **Step 4: Run lifecycle tests to verify they pass**

Run: `python3 -m unittest tests.test_hybrid_knowledge_mcp.PageIndexLifecycleTests -v`

Expected: PASS.

### Task 3: Publish the Confluence-only MCP API and retrieval output

**Files:**
- Modify: `hybrid_knowledge_mcp.py: formatter functions, TOOL_DEFINITIONS, SERVER_INSTRUCTIONS, build_server, selftest`
- Test: `tests/test_hybrid_knowledge_mcp.py`

**Interfaces:**
- Consumes: `KnowledgeBase.index_confluence_page(...) -> PageIndexReport` from Task 2.
- Produces tools: `index_confluence_page`, `search_confluence`, `clear_index`.

- [ ] **Step 1: Write failing MCP schema/validation and result-format tests**

Assert `index_directory` and `search_knowledge_base` are absent; assert the new index schema requires `page_id`, `url`, `title`, and `content`; test rejection of blank required fields and invalid `top_k`; index a known page then assert `search_confluence` results contain its canonical URL, title and breadcrumb.

- [ ] **Step 2: Run API tests to verify they fail**

Run: `python3 -m unittest tests.test_hybrid_knowledge_mcp.McpApiTests -v`

Expected: FAIL because the old tool list and formatting remain.

- [ ] **Step 3: Implement the three MCP tools and page-oriented formatting**

Replace legacy tool definitions and dispatch. Validate all page fields before entering the storage lock, serialize storage calls under the existing gate, rename search/clear output to page terms, and write server instructions that tell the agent to fetch pages using Confluence MCP before calling the indexing tool. Rewrite `--selftest` to index two synthetic Confluence pages, verify deduplication/search/clear/reopen, and never require live Confluence credentials.

- [ ] **Step 4: Run API and complete server tests to verify they pass**

Run: `python3 -m unittest tests.test_hybrid_knowledge_mcp -v`

Expected: PASS.

### Task 4: Add tested Claude Code and Codex configuration support

**Files:**
- Create: `installer_config.py`
- Create: `tests/test_installer_config.py`
- Modify: `install.sh: arguments, config generation/merge, summary`
- Modify: `claude_mcp_config.example.json`
- Create: `codex_mcp_config.example.toml`

**Interfaces:**
- Produces: `build_server_entry(command: str, server_path: str, home: str) -> dict[str, Any]`.
- Produces: `render_claude_snippet(name: str, entry: Mapping[str, Any]) -> str` and `merge_claude_config(existing: str, name: str, entry: Mapping[str, Any]) -> str`.
- Produces: `render_codex_block(name: str, entry: Mapping[str, Any]) -> str` and `merge_codex_config(existing: str, block: str) -> str` using explicit begin/end markers for the managed server block.

- [ ] **Step 1: Write failing config-rendering and merge tests**

Test initial render, preservation of an unrelated Claude MCP entry, replacement of a prior RAG entry, Codex block insertion/replacement, TOML escaping for paths with spaces/quotes, and idempotence of both merge functions.

- [ ] **Step 2: Run config tests to verify they fail**

Run: `python3 -m unittest tests.test_installer_config -v`

Expected: FAIL because the helper module does not exist.

- [ ] **Step 3: Implement deterministic configuration helpers**

Use only the Python standard library. Render the Claude snippet as JSON; manage the Codex entry as a bounded TOML text block so no TOML writing dependency is needed. Use an atomic temporary-file replacement only after successful parsing/rendering.

- [ ] **Step 4: Run config tests to verify they pass**

Run: `python3 -m unittest tests.test_installer_config -v`

Expected: PASS.

- [ ] **Step 5: Integrate helpers into `install.sh`**

Replace Claude-only flags with required `--client claude|codex` selection (or an interactive TTY prompt), copy both Python files into the installation directory, generate the appropriate snippet and update only the selected config. Rename output to Confluence RAG and end with the agent bootstrap prompt instead of directory-indexing instructions. Keep `--skip-selftest`, `--reinstall`, `--dir` and `--name` behavior.

- [ ] **Step 6: Verify shell syntax and non-mutating CLI paths**

Run: `bash -n install.sh && ./install.sh --help`

Expected: exit 0 and help documents `--client` plus both supported clients; no installation or home configuration change occurs.

### Task 5: Document bootstrap and run end-to-end verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_hybrid_knowledge_mcp.py`
- Test: `tests/test_installer_config.py`

**Interfaces:**
- Consumes: the final tool names and installer arguments from Tasks 3 and 4.

- [ ] **Step 1: Write a failing documentation contract test**

Assert the README names `index_confluence_page`, `search_confluence`, `clear_index`, `--client`, both clients and the root Confluence URL, and does not present `index_directory` as an available tool.

- [ ] **Step 2: Run the documentation contract test to verify it fails**

Run: `python3 -m unittest tests.test_hybrid_knowledge_mcp.ReadmeContractTests -v`

Expected: FAIL because the README describes code/directory indexing and Claude Code only.

- [ ] **Step 3: Update documentation and configuration examples**

Describe the agent-mediated crawl: resolve the root, paginate all descendants via Confluence MCP, fetch each page in Markdown, call `index_confluence_page`, report `processed/discovered/indexed/unchanged/failed`, then search. Clearly distinguish explicit reset and targeted page refresh from autonomous search.

- [ ] **Step 4: Run the complete regression suite**

Run: `python3 -m unittest discover -s tests -v && python3 hybrid_knowledge_mcp.py --selftest`

Expected: all tests pass and the isolated self-test reports successful index, deduplication, search, clear, reindex and reopen.

- [ ] **Step 5: Run final static checks and inspect the change set**

Run: `bash -n install.sh && git diff --check && git diff --stat && git status --short`

Expected: no shell syntax or whitespace errors; only intended source, tests, documentation and examples are changed; do not stage or commit unless the user asks.

## Self-Review

- Spec coverage: Tasks 1–3 cover page-only persistent RAG and the three MCP operations; Task 4 covers selected-client installation; Task 5 covers bootstrap guidance and full verification.
- Step scan: every implementation task has an explicit red command, minimal implementation target and green command.
- Type consistency: Task 2 defines `index_confluence_page` and `PageIndexReport`; Task 3 consumes the same method and publishes the matching tool name. Task 4's render/merge helper types are the ones used by the shell integration.
- Review focus: every listed failure mode has a named test in its owning task.
- Proportion: the plan states interfaces, test cases and verification commands without transcribing implementation bodies.
