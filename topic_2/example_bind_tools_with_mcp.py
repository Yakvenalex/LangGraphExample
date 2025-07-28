import asyncio
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from faker import Faker
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


class AgentState(TypedDict):
    """Состояние агента, содержащее последовательность сообщений."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
async def get_random_user_name(gender: str) -> str:
    """
    Возвращает случайное мужское или женское имя в зависимости от условия:
    male - мужчина, female - женщина
    """
    faker = Faker("ru_RU")
    gender = gender.lower()
    if gender == "male":
        return f"{faker.first_name_male()} {faker.last_name_male()}"
    return f"{faker.first_name_female()} {faker.last_name_female()}"


# Ваши существующие инструменты
custom_tools = [get_random_user_name]


async def get_all_tools():
    """Получение всех инструментов: ваших + MCP"""
    # Настройка MCP клиента
    mcp_client = MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
            },
            "match_mcp": {
                "transport": "streamable_http",
                "url": "https://mcpserver-yakvenalex.amvera.io/mcp/",
            },
            # "context7": {
            #     "transport": "streamable_http",
            #     "url": "https://mcp.context7.com/mcp",
            # },
        }
    )

    # Получаем MCP инструменты
    mcp_tools = await mcp_client.get_tools()

    # Объединяем ваши инструменты с MCP инструментами
    all_tools = custom_tools + mcp_tools
    return all_tools


llm = ChatDeepSeek(model="deepseek-chat")


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"


async def main():
    # Получаем все инструменты и MCP клиент
    all_tools = await get_all_tools()

    # Привязываем инструменты к LLM
    llm_with_tools = llm.bind_tools(all_tools)

    # Обновляем функцию model_call для использования llm с инструментами
    async def model_call_with_tools(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(
            content="Ты моя система. Ответь на мой вопрос исходя из доступных для тебя инструментов. "
            "У тебя есть как собственные инструменты, так и инструменты для работы с файловой системой через MCP."
        )
        messages = [system_prompt] + list(state["messages"])
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Создание графа
    graph = StateGraph(AgentState)
    graph.add_node("our_agent", model_call_with_tools)

    # ToolNode с всеми инструментами (ваши + MCP)
    graph.add_node("tools", ToolNode(tools=all_tools))

    # Настройка потока
    graph.add_edge(START, "our_agent")
    graph.add_conditional_edges(
        "our_agent",  # От какого узла
        should_continue,  # Функция-решатель
        {  # Карта решений
            "continue": "tools",  # Если "continue" → идем в "tools"
            "end": END,  # Если "end" → завершаем
        },
    )
    graph.add_edge("tools", "our_agent")

    # Компиляция и запуск
    app = graph.compile()

    result = await app.ainvoke(
        {"messages": [HumanMessage(content="Свойство окружности с радиусом 9?")]}
    )

    # Показываем результат
    print("=== Полная история сообщений ===")
    for i, msg in enumerate(result["messages"]):
        print(f"{i+1}. {type(msg).__name__}: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"   Tool calls: {msg.tool_calls}")

    # Финальный ответ
    final_message = result["messages"][-1]
    print(f"\n=== Финальный ответ ===")
    print(final_message.content)


if __name__ == "__main__":
    asyncio.run(main())
