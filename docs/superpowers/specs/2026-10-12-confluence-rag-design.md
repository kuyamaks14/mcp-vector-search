# Confluence RAG for Claude Code and Codex

## Goal

Turn the existing local MCP server into a documentation-only retrieval-augmented
generation (RAG) service. It indexes the complete page tree rooted at
`https://confluence.o3.ru/x/55-DQw` and lets AI agents retrieve relevant
documentation while working in team projects.

The server no longer indexes source code or arbitrary local directories.

## Constraints

- Confluence content is read only through the configured Confluence MCP server.
- A local process and its `install.sh` script cannot invoke another MCP server
  through the agent host. They must not replace Confluence MCP with direct HTTP
  requests or locally stored credentials.
- The installer configures exactly one selected client: Claude Code or Codex.
- The local server communicates over stdio; its stdout stays reserved for
  JSON-RPC.
- The local index persists across agent restarts and is safe to rebuild.

## Architecture

There are three participants:

1. The AI agent host (Claude Code or Codex) owns the Confluence MCP connection.
   It recursively reads the root page and every descendant using
   `confluence_get_page_children` and `confluence_get_page`.
2. The local RAG MCP server owns embeddings, ChromaDB, BM25 state and page
   metadata. It receives page content from the agent; it never calls
   Confluence itself.
3. `install.sh` installs the local RAG server, writes the selected client's MCP
   configuration and produces clear bootstrap instructions. The first agent
   session performs the Confluence crawl and reports progress to the user.

This separation preserves the user's authenticated Confluence MCP session and
avoids copying credentials into the installation directory.

## MCP API

### `index_confluence_page`

The agent calls this tool only after fetching a page via Confluence MCP.

Input:

- `page_id`: stable Confluence page id;
- `url`: canonical page URL;
- `title`: page title;
- `content`: page body converted to Markdown;
- `version`: optional Confluence page version;
- `ancestors`: ordered page titles from the root to the page.

The server chunks the Markdown and writes it under a page-id-derived namespace.
When the content hash and version have not changed, it returns `unchanged` and
does not recompute embeddings. A changed page replaces only its own previous
chunks. The response includes page/chunk counts so the agent can show a
progress bar while crawling.

### `search_confluence`

Input: `query` and optional `top_k` (1..50, default 5).

The server runs semantic search and BM25, merges results using Reciprocal Rank
Fusion, and returns a result's title, page URL, heading path, excerpt and
ranking signals. Tool instructions require the agent to search when team
documentation is relevant to a task.

### `clear_index`

No input. This destructively removes the Chroma collection, BM25 state and
page manifest. The tool description explicitly limits calls to a user's
unambiguous request to reset the documentation index.

## Bootstrap crawl

After installation the agent performs the following operation, with a visible
`processed / discovered / indexed / unchanged / failed` progress report:

1. Fetch the configured root page by its tiny URL through Confluence MCP.
2. Traverse children depth-first or breadth-first, following pagination until
   every descendant is discovered.
3. Fetch each page body in Markdown and submit it to `index_confluence_page`
   with its metadata and ancestor titles.
4. Finish with an explicit summary and guidance that search is ready.

Page-fetch failures are retained as individual failures in the progress summary;
they do not invalidate successfully indexed pages. The agent reports them and
can retry those pages on request.

## Installer

`install.sh` gains a required client selection, either interactively or through
`--client claude` / `--client codex`. It continues to create the virtual
environment, install dependencies, warm the embedding model and run an
isolated self-test.

It writes:

- Claude Code configuration to `~/.claude.json` under `mcpServers`;
- Codex configuration to `~/.codex/config.toml` under the appropriate
  `mcp_servers` table.

The installer writes a per-client configuration snippet before merging it into
the client configuration. Existing unrelated MCP entries are preserved.

## Data model

The manifest's unit of ownership changes from a filesystem path to a Confluence
page id. For each page it records its content digest, version, title, URL,
ancestor path and generated chunk ids. This supports idempotent page updates
without treating a Confluence title rename as a new document.

Chunks contain page metadata and a heading breadcrumb. No local source paths
or Go-symbol metadata remain.

## Error handling and safety

- Inputs are validated: non-empty id/title/url/content, valid `top_k`, bounded
  request sizes and a valid Markdown string.
- Empty pages remain represented in the manifest but generate no chunks.
- Clear is serialized with indexing and search, like the current Chroma access.
- No Confluence credentials, tokens or page bodies are logged.
- Installation failures leave the old client configuration unchanged through
  parse-before-write and an atomic replacement.

## Testing

Tests cover:

1. chunking and indexing a Confluence page with title and ancestor metadata;
2. unchanged-page deduplication and replacement after a version/content change;
3. hybrid retrieval with result links and breadcrumbs;
4. clearing, then indexing and searching again;
5. validation for each MCP tool;
6. installer rendering and merge behavior for both Claude Code and Codex;
7. the existing isolated end-to-end self-test adapted to page documents.

The test suite does not require a live Confluence connection, a real agent or
real credentials. The production bootstrap uses the Confluence MCP supplied to
the agent host.
