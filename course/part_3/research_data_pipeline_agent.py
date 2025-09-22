import asyncio
import csv
import os
import sqlite3
from pathlib import Path
from typing import TypedDict, Sequence, Annotated, List, Optional

import httpx
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage
)
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ---------------------------
# Константы/«песочница»
# ---------------------------
WORKDIR = Path("./workspace").resolve()
WORKDIR.mkdir(exist_ok=True)
DB_PATH = WORKDIR / "data.sqlite"
CSV_PATH = WORKDIR / "dataset.csv"
REPORT_PATH = WORKDIR / "report.md"

# ---------------------------
# Инструменты (tools)
# (можно заменить на MCP-сервера)
# ---------------------------


@tool
async def web_search(query: str) -> List[str]:
    """
    Простейший web-поиск: возвращает список ссылок.
    Для демонстрации используем DuckDuckGo HTML (без ключей).
    В проде замените на MCP server-fetch + ваш провайдер поиска.
    """
    try:
        url = "https://duckduckgo.com/html/"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, data={"q": query})
            # supa-упрощенный парсинг (мок для примера)
            hrefs = []
            for line in r.text.splitlines():
                if 'result__a' in line and 'href=' in line:
                    # уберём грязь — это демо
                    start = line.find('href="') + 6
                    end = line.find('"', start)
                    link = line[start:end]
                    if link.startswith("http"):
                        hrefs.append(link)
                if len(hrefs) >= 5:
                    break
        return hrefs or ["https://example.com"]
    except Exception as e:
        return [f"ERROR:{e}"]


@tool
async def fetch_url(url: str) -> str:
    """Скачивает HTML/текст по URL (демо: нет JS/рендеринга)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            return r.text[:50_000]  # ограничим объём
    except Exception as e:
        return f"ERROR:{e}"


@tool
async def fs_write_text(path: str, content: str) -> str:
    """
    Пишет текст в файл внутри рабочей директории.
    Безопасность: разрешаем только внутри WORKDIR.
    """
    full = (WORKDIR / Path(path)).resolve()
    if not str(full).startswith(str(WORKDIR)):
        return "ERROR: path outside sandbox"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return f"OK: wrote {full}"


@tool
async def csv_write_rows(path: str, rows: List[List[str]]) -> str:
    """Создаёт/перезаписывает CSV с переданными строками."""
    full = (WORKDIR / Path(path)).resolve()
    if not str(full).startswith(str(WORKDIR)):
        return "ERROR: path outside sandbox"
    with open(full, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return f"OK: wrote {full}"


@tool
async def csv_read_rows(path: str) -> List[List[str]]:
    """Читает CSV и возвращает список строк."""
    full = (WORKDIR / Path(path)).resolve()
    if not str(full).startswith(str(WORKDIR)) or not full.exists():
        return [["ERROR", "file not found or outside sandbox"]]
    with open(full, newline="", encoding="utf-8") as f:
        return [row for row in csv.reader(f)]


@tool
async def sqlite_execute(db_path: str, sql: str, params: Optional[List] = None) -> str:
    """
    Выполняет SQL (DDL/DML). Возвращает 'rows affected'.
    WARNING: демо-инструмент — в проде ограничивайте список разрешённых таблиц/операций.
    """
    full = (WORKDIR / Path(db_path)).resolve()
    if not str(full).startswith(str(WORKDIR)):
        return "ERROR: path outside sandbox"
    full.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(full)
    try:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        conn.commit()
        return f"OK: {cur.rowcount} rows affected"
    finally:
        conn.close()


@tool
async def sqlite_query(db_path: str, sql: str, params: Optional[List] = None) -> List[List[str]]:
    """SELECT-запрос, возвращает строки как списки строк (для простоты)."""
    full = (WORKDIR / Path(db_path)).resolve()
    if not str(full).startswith(str(WORKDIR)) or not full.exists():
        return [["ERROR", "db not found or outside sandbox"]]
    conn = sqlite3.connect(full)
    try:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        return [[str(x) for x in row] for row in rows]
    finally:
        conn.close()

# ---------------------------
# Состояние оркестратора
# ---------------------------


class OrchestratorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    topic: str
    urls: List[str]
    db_path: str
    csv_path: str
    report_path: str

# ---------------------------
# Подагенты (ReAct)
# ---------------------------


def mk_researcher():
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.2)
    sys = SystemMessage(content=(
        "Ты Исследователь. Твоя задача: найти 3–5 релевантных ссылок, "
        "при необходимости коротко скачать содержимое по 1–2 из них и выдать сжатую сводку. "
        "Всегда используй инструменты web_search и fetch_url при необходимости."
    ))
    tools = [web_search, fetch_url]
    return create_react_agent(model=model, tools=tools, prompt=sys, checkpointer=MemorySaver())


def mk_data_engineer():
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
    sys = SystemMessage(content=(
        "Ты Дата-инженер. Тебе дают сводку/факты. "
        "Сформируй таблицу CSV (заголовки + строки), запиши её, "
        "создай/обнови таблицу в SQLite и вставь данные. "
        "Всегда используй csv_write_rows, sqlite_execute, sqlite_query по необходимости."
    ))
    tools = [csv_write_rows, sqlite_execute, sqlite_query]
    return create_react_agent(model=model, tools=tools, prompt=sys, checkpointer=MemorySaver())


def mk_writer():
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
    sys = SystemMessage(content=(
        "Ты Редактор. Получив данные (сводку, CSV, SQL-выборки), создай читабельный отчёт в Markdown. "
        "Сохрани результат на диск через fs_write_text и верни краткое резюме."
    ))
    tools = [fs_write_text, csv_read_rows, sqlite_query]
    return create_react_agent(model=model, tools=tools, prompt=sys, checkpointer=MemorySaver())


researcher = mk_researcher()
data_eng = mk_data_engineer()
writer = mk_writer()

# ---------------------------
# Узлы оркестратора
# ---------------------------


async def node_research(state: OrchestratorState):
    msgs = [
        HumanMessage(content=f"Тема исследования: {state['topic']}. "
                             f"Найди 3–5 ссылок и сделай короткую сводку. Если надо — подтяни содержимое.")
    ]
    res = await researcher.ainvoke({"messages": msgs}, config={"configurable": {"thread_id": "research"}})
    # очень простой парсинг ссылок из ответа (в реале делайте строгий JSON через response_format)
    text = res["messages"][-1].content
    urls = [u for u in text.split() if u.startswith("http")][:5]
    return {"messages": res["messages"], "urls": urls}


async def node_data(state: OrchestratorState):
    # Подготовим задание: на основе сводки (из истории сообщений) попросим сделать CSV + залить в SQLite
    msgs = [
        HumanMessage(content=(
            "На основе предыдущей сводки/фактов сформируй таблицу с колонками "
            "[source, insight] и 3–8 строк. "
            f"Запиши CSV в {state['csv_path']}. "
            f"Далее создай таблицу sales_insights(source TEXT, insight TEXT) в БД {state['db_path']} "
            "и вставь все строки. Верни выборку COUNT(*) для проверки."
        ))
    ]
    res = await data_eng.ainvoke({"messages": msgs}, config={"configurable": {"thread_id": "data"}})
    return {"messages": res["messages"]}


async def node_write(state: OrchestratorState):
    msgs = [
        HumanMessage(content=(
            f"Собери читабельный отчёт в Markdown по теме '{state['topic']}'. "
            f"Используй данные из CSV {state['csv_path']} и выборку из SQLite {state['db_path']} "
            f"(сделай запрос COUNT(*) из sales_insights). "
            f"Сохрани отчёт в {state['report_path']} через fs_write_text. "
            "Верни короткое резюме и путь к файлу."
        ))
    ]
    res = await writer.ainvoke({"messages": msgs}, config={"configurable": {"thread_id": "write"}})
    return {"messages": res["messages"]}

# ---------------------------
# Сборка графа
# ---------------------------
graph = StateGraph(OrchestratorState)
graph.add_node("research", node_research)
graph.add_node("data",     node_data)
graph.add_node("write",    node_write)

graph.set_entry_point("research")
graph.add_edge("research", "data")
graph.add_edge("data", "write")

app = graph.compile(checkpointer=MemorySaver())

# ---------------------------
# Запуск демо
# ---------------------------


async def main():
    # начальное состояние оркестратора
    init = {
        "messages": [HumanMessage(content="Старт")],
        "topic": "_dyn_: влияние погодных условий на продажи кофе в Нидерландах",
        "urls": [],
        "db_path": str(DB_PATH.name),
        "csv_path": str(CSV_PATH.name),
        "report_path": str(REPORT_PATH.name),
    }
    config = {"configurable": {"thread_id": "orchestrator-demo"}}

    result = await app.ainvoke(init, config=config)

    print("\n==== ФИНАЛЬНЫЕ СООБЩЕНИЯ ====")
    for m in result["messages"][-6:]:
        role = type(m).__name__.replace("Message", "")
        print(f"[{role}] {getattr(m,'content','')[:300]}")

    print("\nФайлы в рабочей папке:")
    for p in sorted(WORKDIR.glob("*")):
        print(" -", p.name)

if __name__ == "__main__":
    asyncio.run(main())
