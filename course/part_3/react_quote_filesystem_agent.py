from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_amvera import AmveraLLM
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os
import asyncio
from langchain_deepseek import ChatDeepSeek
import aiohttp
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


# Загрузка переменных окружения
load_dotenv()


class Quote(BaseModel):
    quote: str
    author: str


class AgentState(TypedDict):
    """Состояние агента, содержащее последовательность сообщений."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
async def get_quote() -> Quote:
    """Получить случайную мотивационную цитату в формате {"quote": "цитата", "author": "автор"}."""
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "method": "getQuote",
                "format": "json",
                "lang": "ru"
            }

            async with session.get(
                "https://api.forismatic.com/api/1.0/",
                params=params,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                # Пробуем декодировать как JSON
                data = await response.json()
                print(data)
                quote = data.get("quoteText", "").strip()
                author = data.get("quoteAuthor", "").strip()

                if quote:
                    return {"quote": quote, "author": author}
                else:
                    return {"quote": "Работа не волк. Никто не волк. Только волк — волк.",
                            "author": "Джейсон Стетхем"}
    except Exception as e:
        print(f"Ошибка при получении цитаты: {e}")
        return {"quote": "Если закрыть глаза, становится темно.", "author": "Джейсон Стетхем"}


async def get_all_tools():
    """Получение всех инструментов: ваших + MCP"""
    # Настройка MCP клиента
    mcp_client = MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
            }
        }
    )

    # Получаем MCP инструменты
    mcp_tools = await mcp_client.get_tools()

    # Объединяем ваши инструменты с MCP инструментами
    return [get_quote] + mcp_tools


async def main_react():
    # Получаем все инструменты
    tools = await get_all_tools()

    # Создаём ReAct агента (вся логика графа уже встроена!)
    agent = create_react_agent(
        model=ChatDeepSeek(model="deepseek-chat"),
        tools=tools,
        prompt="В твоём распоряжении есть инструменты для работы с файловой системой и получения мотивационных цитат."
    )

    # Запускаем задачу
    result = await agent.ainvoke({
        "messages": [
            HumanMessage(
                content="Найди мотивационную цитату и сохрани её в файл quote_react.txt с подробной информацией об авторе"
            )
        ]
    })

    print("=== Полная история сообщений ===")
    for i, msg in enumerate(result["messages"]):
        print(f"{i+1}. {type(msg).__name__}: {getattr(msg, 'content', None)}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"   Tool calls: {msg.tool_calls}")


if __name__ == "__main__":
    asyncio.run(main_react())
