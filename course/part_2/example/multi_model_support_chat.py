from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_amvera import AmveraLLM
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()

# Инициализация трех разных моделей
deepseek_model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.1  # Низкая температура для технических задач
)

amvera_model = AmveraLLM(
    model="llama70b",
    temperature=0.7  # Умеренная температура для диалогов
)

gigachat_model = GigaChat(
    model="GigaChat-2-Max",
    temperature=0.3,  # Средняя температура
    verify_ssl_certs=False
)


class TaskClassification(BaseModel):
    task_type: Literal["code", "dialog", "local"] = Field(
        description="Тип задачи: code - программирование, dialog - общение, local - российские реалии"
    )
    confidence: float = Field(
        description="Уверенность в классификации от 0.0 до 1.0",
        ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Краткое объяснение выбора",
        max_length=100
    )


class MultiModelState(TypedDict):
    user_question: str           # Вопрос пользователя
    task_type: str              # Результат классификации
    code_analysis: str          # Результат от DeepSeek
    dialog_response: str        # Результат от Amvera
    local_context: str          # Результат от GigaChat
    final_answer: str           # Итоговый ответ
    should_continue: bool       # Продолжать работу


classification_parser = JsonOutputParser(pydantic_object=TaskClassification)
classification_prompt = PromptTemplate(
    template="""Определи тип задачи пользователя:

CODE - вопросы про программирование, отладку, код, алгоритмы, технологии
DIALOG - обычные вопросы, просьбы о помощи, общение, объяснения
LOCAL - вопросы про Россию, российские законы, локальные особенности, госуслуги

Вопрос: {question}

{format_instructions}

Верни ТОЛЬКО JSON!""",
    input_variables=["question"],
    partial_variables={
        "format_instructions": classification_parser.get_format_instructions()}
)


def classify_task_node(state: MultiModelState) -> dict:
    """Узел классификации задачи - используем DeepSeek"""
    question = state["user_question"]

    try:
        print(f"🤔 Классифицирую задачу...")

        classification_chain = classification_prompt | deepseek_model | classification_parser
        result = classification_chain.invoke({"question": question})

        task_type = result["task_type"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]

        print(f"📋 Тип: {task_type} ({confidence:.2f}) - {reasoning}")

        return {"task_type": task_type}

    except Exception as e:
        print(f"❌ Ошибка классификации: {e}")
        return {"task_type": "dialog"}  # Fallback к диалогу


def code_analysis_node(state: MultiModelState) -> dict:
    """Узел анализа кода - специализация DeepSeek"""
    question = state["user_question"]

    try:
        print("💻 DeepSeek анализирует код...")

        code_messages = [
            SystemMessage(content="""Ты эксперт-программист. Анализируй код, находи ошибки, 
                         предлагай оптимизации. Отвечай технично и точно."""),
            HumanMessage(content=question)
        ]

        response = deepseek_model.invoke(code_messages)
        analysis = response.content

        print(f"✅ DeepSeek: {analysis[:100]}...")

        return {"code_analysis": analysis}

    except Exception as e:
        print(f"❌ Ошибка DeepSeek: {e}")
        return {"code_analysis": "Ошибка анализа кода"}


def dialog_response_node(state: MultiModelState) -> dict:
    """Узел диалогового общения - сила Amvera LLaMA"""
    question = state["user_question"]

    try:
        print("💬 Amvera ведет диалог...")

        dialog_messages = [
            SystemMessage(content="""Ты дружелюбный помощник. Отвечай развернуто, 
                         объясняй простым языком, будь полезным и понимающим."""),
            HumanMessage(content=question)
        ]

        response = amvera_model.invoke(dialog_messages)
        dialog_answer = response.content

        print(f"✅ Amvera: {dialog_answer[:100]}...")

        return {"dialog_response": dialog_answer}

    except Exception as e:
        print(f"❌ Ошибка Amvera: {e}")
        return {"dialog_response": "Ошибка диалогового ответа"}


def local_context_node(state: MultiModelState) -> dict:
    """Узел локального контекста - экспертиза GigaChat"""
    question = state["user_question"]

    try:
        print("🇷🇺 GigaChat анализирует локальный контекст...")

        local_messages = [
            SystemMessage(content="""Ты эксперт по России: законы, традиции, особенности, 
                         госуслуги, местная специфика. Давай точную информацию о российских реалиях."""),
            HumanMessage(content=question)
        ]

        response = gigachat_model.invoke(local_messages)
        local_info = response.content

        print(f"✅ GigaChat: {local_info[:100]}...")

        return {"local_context": local_info}

    except Exception as e:
        print(f"❌ Ошибка GigaChat: {e}")
        return {"local_context": "Ошибка анализа локального контекста"}


def user_input_node(state: MultiModelState) -> dict:
    """Узел получения вопроса от пользователя"""
    question = input("\n❓ Ваш вопрос: ").strip()

    if question.lower() in ["выход", "quit", "exit", "bye"]:
        return {"should_continue": False}

    return {
        "user_question": question,
        "should_continue": True
    }


def synthesize_answer_node(state: MultiModelState) -> dict:
    """Узел синтеза итогового ответа - используем Amvera для объединения"""
    task_type = state["task_type"]
    question = state["user_question"]

    # Собираем доступные результаты
    results = []

    if state.get("code_analysis"):
        results.append(f"Технический анализ: {state['code_analysis']}")

    if state.get("dialog_response"):
        results.append(f"Общий ответ: {state['dialog_response']}")

    if state.get("local_context"):
        results.append(f"Локальная информация: {state['local_context']}")

    if not results:
        return {"final_answer": "Не удалось получить ответ от моделей"}

    try:
        print("🔄 Синтезирую итоговый ответ...")

        synthesis_prompt = f"""На основе результатов от разных ИИ-моделей дай пользователю единый полезный ответ.

Вопрос пользователя: {question}
Тип задачи: {task_type}

Результаты от моделей:
{chr(10).join(results)}

Создай связный, полезный ответ, объединив лучшее из каждого источника."""

        synthesis_messages = [
            SystemMessage(
                content="Ты синтезируешь ответы от разных ИИ в единый полезный ответ."),
            HumanMessage(content=synthesis_prompt)
        ]

        response = amvera_model.invoke(synthesis_messages)
        final_answer = response.content

        print("="*60)
        print("🎯 ИТОГОВЫЙ ОТВЕТ:")
        print("="*60)
        print(final_answer)
        print("="*60)

        return {"final_answer": final_answer}

    except Exception as e:
        print(f"❌ Ошибка синтеза: {e}")
        return {"final_answer": "Ошибка при создании итогового ответа"}


def route_after_input(state: MultiModelState) -> str:
    """Маршрутизация после ввода"""
    if not state.get("should_continue", True):
        return "end"
    return "classify"


def route_after_classification(state: MultiModelState) -> str:
    """Маршрутизация по типу задачи"""
    task_type = state.get("task_type", "dialog")

    if task_type == "code":
        return "analyze_code"
    elif task_type == "local":
        return "local_context"
    else:
        return "dialog_response"


def route_to_synthesis(state: MultiModelState) -> str:
    """Маршрутизация к синтезу ответа"""
    return "synthesize"


def route_continue(state: MultiModelState) -> str:
    """Проверка продолжения"""
    return "get_input" if state.get("should_continue", True) else "end"


# Создание графа
graph = StateGraph(MultiModelState)

# Добавляем узлы
graph.add_node("get_input", user_input_node)
graph.add_node("classify", classify_task_node)
graph.add_node("analyze_code", code_analysis_node)
graph.add_node("dialog_response", dialog_response_node)
graph.add_node("local_context", local_context_node)
graph.add_node("synthesize", synthesize_answer_node)

# Создаем рёбра
graph.add_edge(START, "get_input")

# Условные рёбра
graph.add_conditional_edges(
    "get_input",
    route_after_input,
    {
        "classify": "classify",
        "end": END
    }
)

graph.add_conditional_edges(
    "classify",
    route_after_classification,
    {
        "analyze_code": "analyze_code",
        "dialog_response": "dialog_response",
        "local_context": "local_context"
    }
)

# Все специализированные узлы ведут к синтезу
graph.add_conditional_edges(
    "analyze_code",
    route_to_synthesis,
    {"synthesize": "synthesize"}
)

graph.add_conditional_edges(
    "dialog_response",
    route_to_synthesis,
    {"synthesize": "synthesize"}
)

graph.add_conditional_edges(
    "local_context",
    route_to_synthesis,
    {"synthesize": "synthesize"}
)

graph.add_conditional_edges(
    "synthesize",
    route_continue,
    {
        "get_input": "get_input",
        "end": END
    }
)

# Компиляция
multi_model_app = graph.compile()

if __name__ == "__main__":
    print("🤖 Мультимодельная система техподдержки")
    print("DeepSeek - код | Amvera - диалоги | GigaChat - локальный контекст")
    print("Команда 'выход' для завершения")
    print("-" * 70)

    initial_state = {
        "user_question": "",
        "task_type": "",
        "code_analysis": "",
        "dialog_response": "",
        "local_context": "",
        "final_answer": "",
        "should_continue": True
    }

    try:
        final_state = multi_model_app.invoke(initial_state)
        print("\n✅ Система завершена!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Работа прервана (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Ошибка системы: {e}")
