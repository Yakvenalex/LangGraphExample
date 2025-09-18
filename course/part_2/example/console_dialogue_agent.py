from dotenv import load_dotenv
from langchain_amvera import AmveraLLM
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

load_dotenv()

# Состояние для диалога


class ChatState(TypedDict):
    messages: List[BaseMessage]
    should_continue: bool


# Инициализация LLM
llm = AmveraLLM(model="llama70b")


def user_input_node(state: ChatState) -> dict:
    """Узел для получения ввода пользователя"""
    user_input = input("Вы: ")

    # Проверяем команды выхода
    if user_input.lower() in ["выход", "quit", "exit", "пока", "bye"]:
        return {"should_continue": False}

    # Добавляем сообщение пользователя
    new_messages = state["messages"] + [HumanMessage(content=user_input)]
    return {"messages": new_messages, "should_continue": True}


def llm_response_node(state: ChatState) -> dict:
    """Узел для генерации ответа ИИ"""
    # Получаем ответ от LLM
    response = llm.invoke(state["messages"])
    msg_content = response.content
    # Выводим ответ
    print(f"ИИ: {msg_content}")

    # Добавляем ответ в историю
    new_messages = state["messages"] + [AIMessage(content=msg_content)]
    return {"messages": new_messages}


def should_continue(state: ChatState) -> str:
    """Условная функция для определения продолжения диалога"""
    return "continue" if state.get("should_continue", True) else "end"


# Создание графа
graph = StateGraph(ChatState)

# Добавляем узлы
graph.add_node("user_input", user_input_node)
graph.add_node("llm_response", llm_response_node)

# Создаем рёбра
graph.add_edge(START, "user_input")
graph.add_edge("user_input", "llm_response")

# Условное ребро для проверки продолжения
graph.add_conditional_edges(
    "llm_response",
    should_continue,
    {
        "continue": "user_input",  # Возвращаемся к вводу пользователя
        "end": END                 # Завершаем диалог
    }
)

# Компиляция и запуск
app = graph.compile()

if __name__ == "__main__":
    print("Добро пожаловать в чат с ИИ!")
    print("Для выхода введите: выход, quit, exit, пока, или bye")
    print("-" * 50)

    # Начальное состояние с системным сообщением
    initial_state = {
        "messages": [
            SystemMessage(
                content="Ты дружелюбный помощник. Отвечай коротко и по делу.")
        ],
        "should_continue": True
    }

    # Запуск чата
    final_state = app.invoke(initial_state)

    print("-" * 50)
    print("Чат завершён. До свидания!")
    print(f"Всего сообщений в диалоге: {len(final_state['messages'])}")
