# confluence-rag-mcp

Локальный MCP-сервер для векторного поиска только по командной документации
Confluence. Он хранит индекс на компьютере пользователя, а страницы получает
исключительно через уже настроенный у агента Confluence MCP.

## Установка

```bash
./install.sh --client claude
./install.sh --client codex
```

Установщик создаёт окружение, скачивает локальную модель, выполняет
самопроверку и добавляет сервер только в конфигурацию выбранного клиента.
Для Claude Code используется `~/.claude.json`, для Codex —
`~/.codex/config.toml`. Примеры конфигурации: `claude_mcp_config.example.json`
и `codex_mcp_config.example.toml`.

## Первичная индексация

После установки попросите агента обойти дерево, начиная с
`https://confluence.o3.ru/x/55-DQw`. Агент должен через Confluence MCP
постранично получать детей с пагинацией и тела страниц в Markdown, а затем
вызывать `index_confluence_page` для каждой страницы. Во время обхода он
показывает `processed / discovered / indexed / unchanged / failed` и в конце
сообщает итог.

## MCP API

- `index_confluence_page(page_id, url, title, content, version?, ancestors?)` —
  добавляет или обновляет одну уже полученную страницу.
- `search_confluence(query, top_k?)` — гибридный семантический и BM25-поиск.
- `clear_index()` — полное удаление индекса; вызывается только по явной просьбе
  пользователя.

Сервер не хранит учётные данные Confluence и не выполняет HTTP-запросы к нему.
