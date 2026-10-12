#!/usr/bin/env bash
#
# install.sh — полностью автоматическая установка локального MCP-сервера
# поиска по документации Confluence на macOS.
#
# Что делает:
#   1. Проверяет ОС и находит подходящий python3 (>= 3.10).
#   2. Создаёт ~/.local/share/go-confluence-mcp/ и кладёт туда сервер.
#   3. Разворачивает venv, обновляет pip, ставит зависимости одной командой.
#   4. Заранее скачивает модель all-MiniLM-L6-v2 (холодный старт),
#      чтобы выбранный агент не подвисал на первом поиске.
#   5. Прогоняет сквозную самопроверку: индексация → поиск → очистка.
#   6. Аккуратно прописывает сервер в Claude Code или Codex.
#
# Скрипт идемпотентен: повторный запуск ничего не ломает.
#
# Использование:
#   ./install.sh --client claude     установка для Claude Code
#   ./install.sh --client codex      установка для Codex
#     Codex настраивается в ~/.codex/config.toml и
#     ~/.o3-cli/o3_plugin_agent/codex/config.toml
#   ./install.sh --reinstall         пересоздать venv с нуля
#   ./install.sh --skip-selftest     пропустить самопроверку
#   ./install.sh --dir /custom/path  другой каталог установки

set -Eeuo pipefail

# --------------------------------------------------------------------------- #
# Параметры                                                                    #
# --------------------------------------------------------------------------- #

SERVER_FILE="hybrid_knowledge_mcp.py"
MCP_SERVER_NAME="go-confluence-kb"
INSTALL_DIR="${HOME}/.local/share/go-confluence-mcp"
CLAUDE_CONFIG="${HOME}/.claude.json"
CODEX_CONFIG="${HOME}/.codex/config.toml"
O3_CODEX_CONFIG="${HOME}/.o3-cli/o3_plugin_agent/codex/config.toml"

DO_REINSTALL=0
SKIP_CLAUDE_CONFIG=0
SKIP_SELFTEST=0
CLIENT=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------- #
# Вывод                                                                        #
# --------------------------------------------------------------------------- #

if [ -t 1 ]; then
    C_RESET="$(printf '\033[0m')"
    C_BOLD="$(printf '\033[1m')"
    C_BLUE="$(printf '\033[34m')"
    C_GREEN="$(printf '\033[32m')"
    C_YELLOW="$(printf '\033[33m')"
    C_RED="$(printf '\033[31m')"
    C_DIM="$(printf '\033[2m')"
else
    C_RESET=""; C_BOLD=""; C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_DIM=""
fi

STEP_NO=0
step()  { STEP_NO=$((STEP_NO + 1)); printf '\n%s[%d/8] %s%s\n' "${C_BOLD}${C_BLUE}" "$STEP_NO" "$*" "$C_RESET"; }
info()  { printf '      %s\n' "$*"; }
dim()   { printf '      %s%s%s\n' "$C_DIM" "$*" "$C_RESET"; }
ok()    { printf '      %s✔%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf '      %s!%s %s\n' "$C_YELLOW" "$C_RESET" "$*"; }
die()   { printf '\n%s✘ %s%s\n\n' "$C_RED" "$*" "$C_RESET" >&2; exit 1; }

on_error() {
    local exit_code=$?
    printf '\n%s✘ Установка прервана: строка %s, код выхода %s%s\n' \
        "$C_RED" "${1:-?}" "$exit_code" "$C_RESET" >&2
    printf '  Каталог установки: %s\n' "$INSTALL_DIR" >&2
    printf '  Повторите запуск с флагом --reinstall, если venv повреждён.\n\n' >&2
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

# --------------------------------------------------------------------------- #
# Разбор аргументов                                                            #
# --------------------------------------------------------------------------- #

while [ $# -gt 0 ]; do
    case "$1" in
        --reinstall)          DO_REINSTALL=1; shift ;;
        --skip-claude-config) SKIP_CLAUDE_CONFIG=1; shift ;;
        --skip-selftest)      SKIP_SELFTEST=1; shift ;;
        --client)
            [ $# -ge 2 ] || die "--client требует claude или codex"
            CLIENT="$2"; shift 2 ;;
        --dir)
            [ $# -ge 2 ] || die "--dir требует значение"
            INSTALL_DIR="$2"; shift 2 ;;
        --name)
            [ $# -ge 2 ] || die "--name требует значение"
            MCP_SERVER_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "Неизвестный аргумент: $1 (см. --help)" ;;
    esac
done

if [ "$CLIENT" != "claude" ] && [ "$CLIENT" != "codex" ]; then
    die "укажите --client claude или --client codex"
fi

if [ "$CLIENT" = "claude" ]; then
    CLIENT_LABEL="Claude Code"
    CLIENT_COMMAND="claude"
else
    CLIENT_LABEL="Codex"
    CLIENT_COMMAND="codex"
fi

VENV_DIR="${INSTALL_DIR}/venv"
VENV_PY="${VENV_DIR}/bin/python"
MODELS_DIR="${INSTALL_DIR}/models"
SERVER_PATH="${INSTALL_DIR}/${SERVER_FILE}"

printf '%s\n' "${C_BOLD}════════════════════════════════════════════════════════════════${C_RESET}"
printf '%s\n' "${C_BOLD}  Локальный MCP: гибридный поиск по Go-коду и Confluence${C_RESET}"
printf '%s\n' "${C_BOLD}════════════════════════════════════════════════════════════════${C_RESET}"

# --------------------------------------------------------------------------- #
# 1. Проверка окружения                                                        #
# --------------------------------------------------------------------------- #

step "Проверяю окружение"

OS_NAME="$(uname -s)"
if [ "$OS_NAME" != "Darwin" ]; then
    warn "Скрипт рассчитан на macOS, обнаружено: ${OS_NAME}. Продолжаю на свой риск."
else
    ok "macOS $(sw_vers -productVersion 2>/dev/null || echo '') $(uname -m)"
fi

[ -f "${SCRIPT_DIR}/${SERVER_FILE}" ] \
    || die "Рядом со скриптом нет ${SERVER_FILE} (искал в ${SCRIPT_DIR})"
ok "Исходник сервера найден: ${SCRIPT_DIR}/${SERVER_FILE}"

# Ищем python3 >= 3.10: сначала штатный, затем конкретные версии.
PYTHON_BIN=""
for candidate in python3 python3.13 python3.12 python3.11 python3.10; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    printf '\n%s✘ Не найден Python 3.10+%s\n\n' "$C_RED" "$C_RESET" >&2
    printf '  Установите его, например через Homebrew:\n' >&2
    printf '      brew install python@3.12\n\n' >&2
    exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
ok "Python ${PY_VERSION} → ${PYTHON_BIN}"

if "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 14) else 1)' 2>/dev/null; then
    warn "Python ${PY_VERSION} очень свежий — колёса torch для него могут быть недоступны."
fi

command -v git >/dev/null 2>&1 || warn "git не найден (не критично)"

# --------------------------------------------------------------------------- #
# 2. Каталоги и перенос кода                                                   #
# --------------------------------------------------------------------------- #

step "Готовлю каталог установки"

mkdir -p "${INSTALL_DIR}" "${MODELS_DIR}" "${INSTALL_DIR}/data"
ok "Каталог: ${INSTALL_DIR}"

cp "${SCRIPT_DIR}/${SERVER_FILE}" "${SERVER_PATH}"
cp "${SCRIPT_DIR}/installer_config.py" "${INSTALL_DIR}/installer_config.py"
chmod 0755 "${SERVER_PATH}"
ok "Сервер скопирован: ${SERVER_PATH}"

# --------------------------------------------------------------------------- #
# 3. Виртуальное окружение                                                     #
# --------------------------------------------------------------------------- #

step "Создаю виртуальное окружение"

if [ "$DO_REINSTALL" -eq 1 ] && [ -d "$VENV_DIR" ]; then
    info "Флаг --reinstall: удаляю старый venv"
    rm -rf "$VENV_DIR"
fi

if [ -x "$VENV_PY" ]; then
    ok "venv уже существует: ${VENV_DIR}"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "venv создан: ${VENV_DIR}"
fi

[ -x "$VENV_PY" ] || die "venv собран некорректно, нет ${VENV_PY}. Запустите с --reinstall"

# --------------------------------------------------------------------------- #
# 4. Зависимости                                                               #
# --------------------------------------------------------------------------- #

step "Обновляю pip и ставлю зависимости"

info "Это самый долгий шаг: torch весит ~250 МБ, суммарно нужно ~2.5 ГБ."

"$VENV_PY" -m pip install --upgrade --disable-pip-version-check --quiet pip setuptools wheel
ok "pip $("$VENV_PY" -m pip --version | awk '{print $2}')"

# На Intel-маках сборки torch остановились на 2.2 — не пиним минимум,
# пусть резолвер сам возьмёт последнее доступное колесо.
if [ "$(uname -m)" = "x86_64" ]; then
    TORCH_SPEC="torch"
else
    TORCH_SPEC="torch>=2.2.0"
fi

# mcp 2.x ломает low-level API (переход на сигнатуру (ctx, params)) — держимся 1.x.
DEPS="mcp>=1.9.0,<2.0.0
sentence-transformers>=3.0.0,<6.0.0
chromadb>=0.5.5,<2.0.0
rank-bm25>=0.2.2
${TORCH_SPEC}"

printf '%s\n' "$DEPS" > "${INSTALL_DIR}/requirements.txt"
dim "Список зафиксирован в ${INSTALL_DIR}/requirements.txt"

# Одной командой, как и задумано.
"$VENV_PY" -m pip install --upgrade --disable-pip-version-check \
    "mcp>=1.9.0,<2.0.0" \
    "sentence-transformers>=3.0.0,<6.0.0" \
    "chromadb>=0.5.5,<2.0.0" \
    "rank-bm25>=0.2.2" \
    "$TORCH_SPEC"

ok "Зависимости установлены"

"$VENV_PY" - <<'PY'
import importlib.metadata as md
import sys
for name in ("mcp", "sentence-transformers", "chromadb", "rank-bm25", "torch", "numpy"):
    try:
        print("      %-22s %s" % (name, md.version(name)))
    except md.PackageNotFoundError:
        print("      %-22s ОТСУТСТВУЕТ" % name, file=sys.stderr)
        sys.exit(1)
PY

# --------------------------------------------------------------------------- #
# 5. Холодная загрузка модели                                                  #
# --------------------------------------------------------------------------- #

step "Скачиваю модель all-MiniLM-L6-v2 (холодный старт)"

export HF_HOME="${MODELS_DIR}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export ANONYMIZED_TELEMETRY=False

info "Модель кладётся в ${MODELS_DIR} — ${CLIENT_LABEL} не будет ждать её при первом поиске."

"$VENV_PY" "$SERVER_PATH" --warmup --base-dir "$INSTALL_DIR"

MODEL_SIZE="$(du -sh "${MODELS_DIR}" 2>/dev/null | awk '{print $1}')"
ok "Модель на диске (${MODEL_SIZE:-?})"

# --------------------------------------------------------------------------- #
# 6. Самопроверка                                                              #
# --------------------------------------------------------------------------- #

step "Сквозная самопроверка"

if [ "$SKIP_SELFTEST" -eq 1 ]; then
    warn "Пропущена по флагу --skip-selftest"
else
    info "Индексация → повторная индексация по хэшам → гибридный поиск → очистка."
    if "$VENV_PY" "$SERVER_PATH" --selftest --base-dir "$INSTALL_DIR" 2>&1 | sed 's/^/      /'; then
        ok "Все проверки пройдены"
    else
        die "Самопроверка не прошла — смотрите вывод выше"
    fi
fi

# --------------------------------------------------------------------------- #
# 7. Конфигурация клиента                                                      #
# --------------------------------------------------------------------------- #

step "Настраиваю MCP для ${CLIENT_LABEL}"

if [ "$CLIENT" = "codex" ]; then
    for CODEX_TARGET in "${CODEX_CONFIG}" "${O3_CODEX_CONFIG}"; do
        mkdir -p "$(dirname "${CODEX_TARGET}")"
        "$VENV_PY" - "${SCRIPT_DIR}" "${CODEX_TARGET}" "$MCP_SERVER_NAME" "$VENV_PY" "$SERVER_PATH" "$INSTALL_DIR" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[1])
from installer_config import build_server_entry, merge_codex_config

target, name, command, server, home = sys.argv[2:]
existing = open(target, encoding="utf-8").read() if os.path.exists(target) else ""
content = merge_codex_config(existing, name, build_server_entry(command, server, home))
tmp = target + ".tmp-install"
with open(tmp, "w", encoding="utf-8") as handle:
    handle.write(content)
os.replace(tmp, target)
PY
        ok "Конфигурация Codex обновлена: ${CODEX_TARGET}"
    done
else

CONFIG_SNIPPET="${INSTALL_DIR}/claude_mcp_config.json"
"$VENV_PY" - "$CONFIG_SNIPPET" "$MCP_SERVER_NAME" "$VENV_PY" "$SERVER_PATH" "$INSTALL_DIR" <<'PY'
import json, sys
out, name, command, server, home = sys.argv[1:6]
entry = {
    "type": "stdio",
    "command": command,
    "args": [server],
    "env": {
        "GO_CONFLUENCE_MCP_HOME": home,
        "HF_HOME": home + "/models",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "ANONYMIZED_TELEMETRY": "False",
        "OMP_NUM_THREADS": "4",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    },
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"mcpServers": {name: entry}}, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
dim "Готовый фрагмент конфига: ${CONFIG_SNIPPET}"

if [ "$SKIP_CLAUDE_CONFIG" -eq 1 ]; then
    warn "Пропущено по флагу --skip-claude-config"
else
    if pgrep -x claude >/dev/null 2>&1; then
        warn "Claude Code сейчас запущен — он перезаписывает ~/.claude.json при выходе."
        warn "После установки закройте и откройте его заново."
    fi

    set +e
    MERGE_OUTPUT="$("$VENV_PY" - "$CLAUDE_CONFIG" "$CONFIG_SNIPPET" "$MCP_SERVER_NAME" <<'PY'
import json, os, shutil, sys, time

config_path, snippet_path, name = sys.argv[1:4]

with open(snippet_path, encoding="utf-8") as handle:
    entry = json.load(handle)["mcpServers"][name]

# Пишем по реальному пути, чтобы не заменить симлинк обычным файлом.
target = os.path.realpath(config_path)

data = {}
action = "создан"
if os.path.exists(target) and os.path.getsize(target) > 0:
    try:
        with open(target, encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        print("PARSE_ERROR: %s" % error)
        sys.exit(2)
    if not isinstance(data, dict):
        print("PARSE_ERROR: корень ~/.claude.json не является JSON-объектом")
        sys.exit(2)

    backup = "%s.backup-%s" % (target, time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(target, backup)
    print("BACKUP: %s" % backup)
    action = "обновлён"

servers = data.get("mcpServers")
if not isinstance(servers, dict):
    servers = {}

replaced = name in servers
servers[name] = entry
data["mcpServers"] = servers

tmp = target + ".tmp-install"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(data, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(tmp, target)

print("OK: %s (%s), запись '%s' %s" % (
    target, action, name, "заменена" if replaced else "добавлена"))
print("SERVERS: %s" % ", ".join(sorted(servers)))
PY
)"
    MERGE_STATUS=$?
    set -e

    printf '%s\n' "$MERGE_OUTPUT" | sed 's/^/      /'

    if [ "$MERGE_STATUS" -eq 0 ]; then
        ok "Конфигурация Claude Code обновлена"
    else
        warn "Не удалось разобрать ${CLAUDE_CONFIG} — файл НЕ тронут."
        warn "Добавьте вручную содержимое ${CONFIG_SNIPPET} в блок \"mcpServers\"."
    fi
fi

fi

# --------------------------------------------------------------------------- #
# 8. Итог                                                                      #
# --------------------------------------------------------------------------- #

step "Готово"

cat <<EOF

${C_GREEN}${C_BOLD}Установка завершена.${C_RESET}

  Сервер   : ${SERVER_PATH}
  Python   : ${VENV_PY}
  Индексы  : ${INSTALL_DIR}/data
  Модель   : ${MODELS_DIR}

${C_BOLD}Что делать дальше${C_RESET}

  1. Откройте терминал в вашем проекте и запустите ${CLIENT_LABEL}:

         ${C_BOLD}${CLIENT_COMMAND}${C_RESET}

  2. В первой же сессии попросите ${CLIENT_LABEL} получить страницы через
     Confluence MCP и передать их в index_confluence_page.

  3. Дальше просто задавайте вопросы — ${CLIENT_LABEL} вызовет
     search_confluence и получит фрагменты документации.

${C_DIM}  Полная пересборка индекса: попросите ${CLIENT_LABEL} вызвать clear_index
  Логи сервера пишутся в stderr и видны в выводе ${CLIENT_LABEL}.${C_RESET}

EOF
