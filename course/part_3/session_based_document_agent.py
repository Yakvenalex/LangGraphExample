import asyncio
import sys
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


@tool
async def session_status() -> str:
    """Показывает статус текущей сессии и доступные файлы."""
    return "Сессия активна. Используйте filesystem инструменты для работы с документами."


@tool
async def end_session(reason: str = "Пользователь завершил работу") -> str:
    """Завершает текущую сессию работы с документами."""
    print(f"\n🔚 Завершение сессии: {reason}")
    return f"Сессия завершена. {reason}"


async def get_all_tools():
    """Получение всех инструментов: MCP + управление сессией."""
    custom_tools = [session_status, end_session]

    try:
        mcp_client = MultiServerMCPClient({
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
            }
        })
        mcp_tools = await mcp_client.get_tools()
        print(f"📁 Подключено {len(mcp_tools)} инструментов filesystem")
        return custom_tools + mcp_tools
    except Exception as e:
        print(f"⚠️  MCP недоступен, используем базовые инструменты: {e}")
        return custom_tools


async def create_agent():
    """Создание агента с инструментами."""
    tools = await get_all_tools()

    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.3,
    )

    system_prompt = """Ты профессиональный помощник по созданию и редактированию писем и документов.

У тебя есть полный доступ к файловой системе через MCP инструменты:
- read_file, write_file - чтение и запись файлов
- list_directory - просмотр содержимого папок  
- create_directory - создание папок
- move_file, copy_file - операции с файлами

ПРИНЦИПЫ РАБОТЫ:
1. Помогай пользователю создавать качественные письма и документы
2. Всегда сохраняй результаты работы в файлы
3. Предлагай улучшения и редактирование
4. Поддерживай контекст сессии - помни о созданных файлах и документах

ЗАВЕРШЕНИЕ СЕССИИ:
Используй end_session() когда:
- Пользователь явно просит завершить ("закончить", "выйти", "хватит")
- Работа полностью выполнена и пользователь доволен результатом
- После фраз типа "спасибо", "готово", "всё хорошо"

ВАЖНО: Будь полезным, дружелюбным и профессиональным!"""

    checkpointer = MemorySaver()
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        prompt=SystemMessage(content=system_prompt)
    )

    print("✅ Агент инициализирован с персистентной памятью")
    return agent


async def print_session_stats(agent, config):
    """Выводит статистику сессии."""
    try:
        state = await agent.aget_state(config)
        if state and state.values.get("messages"):
            message_count = len(state.values["messages"])
            print(f"💬 Обработано сообщений: {message_count}")
            print(f"🆔 ID потока: {config['configurable']['thread_id']}")
        else:
            print("📊 Статистика недоступна")
    except Exception as e:
        print(f"⚠️  Ошибка при получении статистики: {e}")


async def run_interactive_session():
    """Запуск интерактивной сессии."""
    print("🤖 ИНТЕРАКТИВНЫЙ ПОМОЩНИК ПО СОЗДАНИЮ ПИСЕМ")
    print("💡 Команды: 'выход', 'quit', 'стоп' - для завершения")
    print("📝 Просто опишите что нужно создать или отредактировать")

    # Создаем агента
    agent = await create_agent()
    config = {"configurable": {"thread_id": "document-session"}}

    try:
        # Основной цикл
        while True:
            try:
                user_input = input("\n👤 Ваш запрос: ").strip()

                if not user_input:
                    continue

                # Проверяем команды выхода
                if user_input.lower() in ['выход', 'quit', 'exit', 'стоп', 'stop']:
                    print("\n👋 До свидания!")
                    break

                print("\n🔄 Обрабатываю запрос...")

                # Отправляем сообщение агенту
                user_message = HumanMessage(content=user_input)
                response_printed = False
                session_ended = False

                async for chunk in agent.astream({"messages": [user_message]}, config=config, stream_mode="messages"):
                    if "messages" in chunk and chunk["messages"]:
                        last_msg = chunk["messages"][-1]

                        if isinstance(last_msg, AIMessage) and not response_printed:
                            print(f"\n🤖 {last_msg.content}")
                            response_printed = True

                            # Проверяем завершение сессии
                            if last_msg.tool_calls:
                                for tool_call in last_msg.tool_calls:
                                    if tool_call["name"] == "end_session":
                                        session_ended = True

                if session_ended:
                    print("🔚 Агент завершил сессию")
                    break

            except KeyboardInterrupt:
                print("\n\n⚠️  Получен сигнал прерывания")
                break
            except Exception as e:
                print(f"\n❌ Ошибка при обработке запроса: {e}")
                continue

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
    finally:
        print("\n" + "=" * 60)
        print("📊 СТАТИСТИКА СЕССИИ")
        await print_session_stats(agent, config)
        print("🏁 Сессия завершена")


async def main():
    """Главная функция приложения."""
    try:
        await run_interactive_session()
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем")
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
