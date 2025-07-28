# пример клиента для тестирования основного mcp сервера
import asyncio
import json
from fastmcp import Client
from dotenv import load_dotenv
import os

load_dotenv()


async def test_math_server():
    """Простое тестирование математического MCP сервера."""

    print("🧮 Подключаемся к математическому серверу...")
    client = Client(os.getenv("BASE_MCP_URL"))

    async with client:
        try:
            # Проверяем соединение
            await client.ping()
            print("✅ Сервер работает!\n")

            # Получаем список возможностей
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

            # Показываем что доступно
            print(f"📋 Доступно инструментов: {len(tools)}")
            for tool in tools:
                print(f"  • {tool.name}")

            print(f"\n📚 Доступно ресурсов: {len(resources)}")
            for resource in resources:
                print(f"  • {resource.uri}")

            print(f"\n💭 Доступно промптов: {len(prompts)}")
            for prompt in prompts:
                print(f"  • {prompt.name}")

            print("\n🧪 ТЕСТИРУЕМ ОСНОВНЫЕ ФУНКЦИИ:")
            print("-" * 40)

            # Вспомогательная функция для извлечения данных
            def extract_data(result):
                """Извлечь данные из MCP результата."""
                if isinstance(result.content, list) and len(result.content) > 0:
                    text_content = result.content[0]
                    if hasattr(text_content, 'text'):
                        return json.loads(text_content.text)
                return result.content

            # 1. Базовое вычисление
            print("1️⃣ Базовое вычисление:")
            result = await client.call_tool("calculate_basic", {"expression": "2 + 3 * 4"})
            data = extract_data(result)
            print(f"   2 + 3 * 4 = {data['result']}")

            # 2. Квадратное уравнение
            print("\n2️⃣ Квадратное уравнение x² - 5x + 6 = 0:")
            result = await client.call_tool("solve_quadratic", {"a": 1, "b": -5, "c": 6})
            data = extract_data(result)
            print(f"   Корни: {data['roots']}")

            # 3. Факториал
            print("\n3️⃣ Факториал 5:")
            result = await client.call_tool("factorial", {"n": 5})
            data = extract_data(result)
            print(f"   5! = {data['factorial']}")

            # 4. Геометрия - окружность
            print("\n4️⃣ Свойства окружности с радиусом 3:")
            result = await client.call_tool("circle_properties", {"radius": 3})
            data = extract_data(result)
            print(f"   Площадь: {data['area']:.2f}")
            print(f"   Длина: {data['circumference']:.2f}")

            # 5. Статистика
            print("\n5️⃣ Анализ данных [1, 2, 3, 4, 5]:")
            result = await client.call_tool("analyze_dataset", {"numbers": [1, 2, 3, 4, 5]})
            data = extract_data(result)
            print(f"   Среднее: {data['mean']}")
            print(f"   Медиана: {data['median']}")

            # 6. Читаем ресурс - формулы
            print("\n6️⃣ Читаем математические формулы:")
            resource = await client.read_resource("math://formulas/basic")
            print(f"   DEBUG: resource = {resource}")
            print(f"   DEBUG: type = {type(resource)}")
            # Пробуем разные способы доступа к ресурсу
            if hasattr(resource, 'content'):
                print(f"   Получили {len(resource.content)} символов формул")
            elif isinstance(resource, list) and len(resource) > 0:
                content = resource[0]
                if hasattr(content, 'text'):
                    print(f"   Получили {len(content.text)} символов формул")
                else:
                    print(f"   Ресурс: {content}")
            else:
                print(f"   Ресурс: {resource}")

            # 7. Генерируем промпт
            print("\n7️⃣ Генерируем промпт для объяснения:")
            prompt = await client.get_prompt("explain_solution", {
                "problem": "x² - 4 = 0",
                "solution": "x = ±2",
                "level": "beginner"
            })
            print(f"   DEBUG: prompt = {type(prompt)}")
            if hasattr(prompt, 'content'):
                print(
                    f"   Создан промпт длиной {len(prompt.content)} символов")
            else:
                print(f"   Промпт: {prompt}")

            print("\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_math_server())
