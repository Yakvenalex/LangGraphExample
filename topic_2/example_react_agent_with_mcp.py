import asyncio
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from faker import Faker
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


load_dotenv()

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
    return  [get_random_user_name] + mcp_tools


async def run_query(agent, query: str):
    """Выполняет один запрос к агенту с читаемым выводом"""
    print(f"🎯 Запрос: {query}")

    step_counter = 0
    processed_messages = set()  # Для избежания дублирования
    
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        if "messages" in event and event["messages"]:
            messages = event["messages"]
            
            # Обрабатываем только новые сообщения
            for msg in messages:
                msg_id = getattr(msg, 'id', str(id(msg)))
                if msg_id in processed_messages:
                    continue
                processed_messages.add(msg_id)
                
                # Получаем тип сообщения
                msg_type = getattr(msg, 'type', 'unknown')
                content = getattr(msg, 'content', '')
                
                # 1. Сообщения от пользователя
                if msg_type == 'human':
                    print(f"👤 Пользователь: {content}")
                    print("-" * 40)
                
                # 2. Сообщения от ИИ
                elif msg_type == 'ai':
                    # Проверяем наличие вызовов инструментов
                    tool_calls = getattr(msg, 'tool_calls', [])
                    
                    if tool_calls:
                        step_counter += 1
                        print(f"🤖 Шаг {step_counter}: Агент использует инструменты")
                        
                        # Размышления агента (если есть)
                        if content and content.strip():
                            print(f"💭 Размышления: {content}")
                        
                        # Детали каждого вызова инструмента
                        for i, tool_call in enumerate(tool_calls, 1):
                            # Парсим tool_call в зависимости от формата
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get('name', 'unknown')
                                tool_args = tool_call.get('args', {})
                                tool_id = tool_call.get('id', 'unknown')
                            else:
                                # Если это объект с атрибутами
                                tool_name = getattr(tool_call, 'name', 'unknown')
                                tool_args = getattr(tool_call, 'args', {})
                                tool_id = getattr(tool_call, 'id', 'unknown')
                            
                            print(f"🔧 Инструмент {i}: {tool_name}")
                            print(f"   📥 Параметры: {tool_args}")
                            print(f"   🆔 ID: {tool_id}")
                        
                        print("-" * 40)
                    
                    # Финальный ответ (без tool_calls)
                    elif content and content.strip():
                        print(f"🎉 Финальный ответ:")
                        print(f"💬 {content}")
                        print("-" * 40)
                
                # 3. Результаты выполнения инструментов
                elif msg_type == 'tool':
                    tool_name = getattr(msg, 'name', 'unknown')
                    tool_call_id = getattr(msg, 'tool_call_id', 'unknown')
                    
                    print(f"📤 Результат инструмента: {tool_name}")
                    print(f"   🆔 Call ID: {tool_call_id}")
                    
                    # Форматируем результат
                    if content:
                        # Пытаемся распарсить JSON для красивого вывода
                        try:
                            import json
                            if content.strip().startswith(('{', '[')):
                                parsed = json.loads(content)
                                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                                print(f"   📊 Результат:")
                                for line in formatted.split('\n'):
                                    print(f"      {line}")
                            else:
                                print(f"   📊 Результат: {content}")
                        except:
                            print(f"   📊 Результат: {content}")
                    
                    print("-" * 40)
                
                # 4. Другие типы сообщений (для отладки)
                else:
                    if content:
                        print(f"❓ Неизвестный тип ({msg_type}): {content[:100]}...")
                        print("-" * 40)
    
    print("=" * 80)
    print("✅ Запрос обработан")
    print()


async def main():
    # Получаем все инструменты и MCP клиент
    all_tools = await get_all_tools()
    agent = create_react_agent(model=ChatDeepSeek(model="deepseek-chat"),
                               tools=all_tools,
                               prompt="Ты дружелюбный ассистент, который может генерировать фейковых пользователей, \
                                   выполнять вычисления и делиться интересными фактами.",
                               )
    await run_query(agent, query="Свойство окружности с радиусом 29 и придумай сженское имя?")
    

asyncio.run(main())