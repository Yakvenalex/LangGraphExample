import asyncio
import random
from typing import TypedDict
from dotenv import load_dotenv
from faker import Faker
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent

# --- Инициализация ---
load_dotenv()
fake = Faker("ru_RU")


# --- Типы данных ---
class FakeUser(TypedDict):
    name: str
    age: int
    job: str
    email: str
    interests: list[str]


# --- Асинхронные инструменты ---
async def generate_fake_user() -> FakeUser:
    """Асинхронно генерирует фейкового пользователя с основными данными"""
    # Имитируем асинхронную работу (например, запрос к API)
    await asyncio.sleep(0.01)

    return {
        "name": fake.name(),
        "age": random.randint(18, 60),
        "job": fake.job(),
        "email": fake.email(),
        "interests": random.sample(
            [
                "чтение",
                "путешествия",
                "спорт",
                "программирование",
                "фотография",
                "кулинария",
                "музыка",
                "йога",
                "настольные игры",
                "рисование",
                "танцы",
                "велосипед",
                "кино",
                "театр",
            ],
            k=3,
        ),
    }


async def get_random_fact() -> str:
    """Возвращает случайный интересный факт"""
    await asyncio.sleep(0.15)

    facts = [
        "Осьминоги имеют три сердца и синюю кровь",
        "Мед никогда не портится - археологи находили съедобный мед возрастом 3000 лет",
        "Бананы технически являются ягодами, а клубника - нет",
        "Акулы старше деревьев - они существуют уже 400 миллионов лет",
        "В космосе металлы могут свариваться друг с другом без нагрева",
    ]
    return random.choice(facts)


# Список инструментов
tools = [generate_fake_user, get_random_fact]


# Создаем ReAct агента с помощью create_react_agent
agent = create_react_agent(
    model=ChatDeepSeek(model="deepseek-chat"),
    tools=tools,
    prompt="Ты дружелюбный ассистент, который может генерировать фейковых пользователей, выполнять вычисления и делиться интересными фактами. Отвечай понятно и с энтузиазмом!",
)


# --- Асинхронные функции ---
async def run_query(query: str):
    """Выполняет один запрос к агенту с сохранением контекста"""
    print(f"🎯 Запрос: {query}")
    print("-" * 60)

    # Используем astream для асинхронного стримингового выполнения
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",  # Получаем полные значения состояния
    ):
        # Выводим только последнее сообщение от агента
        if "messages" in event and event["messages"]:
            last_message = event["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                # Проверяем, что это сообщение от ассистента
                if hasattr(last_message, "type") and last_message.type == "ai":
                    print(f"🤖 Агент: {last_message.content}")
                    print()


async def run_conversation(
    query="Привет! Создай мне фейкового пользователя и расскажи о нем подробно",
):
    """Запускает полноценный разговор с агентом"""
    print("🚀 Запуск асинхронного LangGraph агента...\n")
    await run_query(query)


async def main():
    """Главная асинхронная функция"""
    try:
        # Основной разговор с сохранением контекста
        await run_conversation()
        print("\n✅ Демонстрация завершена!")
    except KeyboardInterrupt:
        print("\n🛑 Работа прервана пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


# --- Запуск ---
if __name__ == "__main__":
    print("🎯 Запуск обновленного асинхронного LangGraph агента (2025)...")
    asyncio.run(main())
