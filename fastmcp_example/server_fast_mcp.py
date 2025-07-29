from fastmcp import FastMCP
import json
import datetime

# Создаем сервер
mcp = FastMCP(
    name="Demo Assistant",
    instructions="Ассистент для демонстрации возможностей MCP"
)



@mcp.tool
def calculate_age(birth_year: int) -> int:
    """Вычисляет возраст по году рождения"""
    current_year = datetime.datetime.now().year
    return current_year - birth_year


@mcp.tool
async def generate_password(length: int = 12) -> str:
    """Генерирует случайный пароль"""
    import random
    import string
    chars = string.ascii_letters + string.digits + "!@#$%"
    return ''.join(random.choice(chars) for _ in range(length))

# === РЕСУРСЫ ===


@mcp.resource("system://status")
def system_status() -> str:
    """Возвращает статус системы"""
    return json.dumps({
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0.0"
    })


@mcp.resource("help://{topic}")
def get_help(topic: str) -> str:
    """Возвращает справку по теме"""
    help_docs = {
        "password": "Используйте generate_password для создания безопасных паролей",
        "age": "Используйте calculate_age для вычисления возраста",
        "status": "Проверьте system://status для мониторинга системы"
    }
    return help_docs.get(topic, f"Справка по теме '{topic}' не найдена")


@mcp.prompt
def security_check(action: str) -> str:
    """Создает промпт для проверки безопасности действия"""
    return f"""
    Ты специалист по информационной безопасности. 
    Проанализируй это действие на предмет безопасности: {action}
    
    Оцени:
    1. Потенциальные риски
    2. Рекомендации по безопасности  
    3. Альтернативные подходы
    """


@mcp.prompt
def explain_result(tool_name: str, result: str) -> str:
    """Объясняет результат работы инструмента"""
    return f"""
    Объясни пользователю простыми словами результат работы инструмента '{tool_name}':
    
    Результат: {result}
    
    Сделай объяснение понятным и полезным.
    """


# Запуск сервера
if __name__ == "__main__":
    mcp.run(transport="http", port=8099)
