from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os
import asyncio

# Загрузка переменных окружения
load_dotenv()


class AgentState(TypedDict):
    """Состояние агента, содержащее последовательность сообщений."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
async def add(a: int, b: int) -> int:
    """Складывает два целых числа и возвращает результат."""
    await asyncio.sleep(0.1)
    return a + b


@tool
async def list_files() -> list:
    """Возвращает список файлов в текущей папке."""
    await asyncio.sleep(0.1)
    return os.listdir(".")


tools = [add, list_files]

# Инициализация модели чата
llm = ChatDeepSeek(model="deepseek-chat").bind_tools(tools)


async def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="Ты моя система. Ответь на мой вопрос исходя из доступных для тебя инструментов"
    )
    messages = [system_prompt] + list(state["messages"])
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


async def should_continue(state: AgentState) -> str:
    """Проверяет, нужно ли продолжить выполнение или закончить."""
    messages = state["messages"]
    last_message = messages[-1]

    # Если последнее сообщение от AI и содержит вызовы инструментов - продолжаем
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"

    # Иначе заканчиваем
    return "end"


async def main():
    # Создание графа
    graph = StateGraph(AgentState)
    graph.add_node("our_agent", model_call)
    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    # Настройка потока
    graph.add_edge(START, "our_agent")
    graph.add_conditional_edges(
        "our_agent", should_continue, {"continue": "tools", "end": END}
    )
    graph.add_edge("tools", "our_agent")

    # Компиляция и запуск
    app = graph.compile()
    result = await app.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Посчитай общее количество файлов в этой дирректории и прибавь к этому значению 10"
                )
            ]
        }
    )

    # Показываем результат
    print("=== Полная история сообщений ===")
    for i, msg in enumerate(result["messages"]):
        print(f"{i+1}. {type(msg).__name__}: {getattr(msg, 'content', None)}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"   Tool calls: {msg.tool_calls}")

    # Финальный ответ
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            print(f"\n=== Финальный ответ ===")
            print(msg.content)
            break
    else:
        print("\n=== Финальный ответ не найден ===")


if __name__ == "__main__":
    asyncio.run(main())
