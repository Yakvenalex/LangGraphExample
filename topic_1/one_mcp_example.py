import asyncio
import logging
import os
from enum import Enum
from functools import wraps
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


# ===== НАСТРОЙКА ЛОГИРОВАНИЯ =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # Упрощен формат
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_agent.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ===== ЕНУМЫ И КОНСТАНТЫ =====
class ModelProvider(Enum):
    """Доступные провайдеры моделей"""
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"



# ===== ДЕКОРАТОРЫ =====
def retry_on_failure(max_retries: int = 2, delay: float = 1.0):
    """Декоратор для повторения операций при неудаче"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Попытка {attempt + 1} неудачна, повтор через {delay}с")
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


# ===== УПРОЩЕННАЯ КОНФИГУРАЦИЯ =====
@dataclass
class AgentConfig:
    """Упрощенная конфигурация AI-агента"""
    filesystem_path: str = "/home/yakvenalex/PythonProjects/ai_agent_project"
    model_provider: ModelProvider = ModelProvider.OLLAMA
    use_memory: bool = True
    
    # Упрощенные настройки моделей
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "ollama": {
            "model_name": "qwen2.5:32b",
            "base_url": "http://localhost:11434",
            "temperature": 0.0
        },
        "openrouter": {
            "model_name": "moonshotai/kimi-k2:free",
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.0
        },
        "openai": {
            "model_name": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "temperature": 0.0
        }
        ,
        "deepseek": {
            "model_name": "deepseek-chat",
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0.0
        }
    })
    
    def validate(self) -> None:
        """Простая валидация"""
        if not os.path.exists(self.filesystem_path):
            raise ValueError(f"Путь не существует: {self.filesystem_path}")
        
        config = self.model_configs.get(self.model_provider.value)
        if not config:
            raise ValueError(f"Неподдерживаемый провайдер: {self.model_provider}")
        
        # Проверка API ключа если нужен
        api_key_env = config.get("api_key_env")
        if api_key_env and not os.getenv(api_key_env):
            raise ValueError(f"Отсутствует переменная окружения: {api_key_env}")
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Конфигурация MCP сервера"""
        return {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", self.filesystem_path],
                "transport": "stdio"
            }
        }


# ===== УПРОЩЕННАЯ ФАБРИКА МОДЕЛЕЙ =====
class ModelFactory:
    """Упрощенная фабрика моделей"""
    
    @staticmethod
    def create_model(config: AgentConfig):
        """Создает модель согласно конфигурации"""
        provider = config.model_provider.value
        model_config = config.model_configs[provider]
        
        logger.info(f"Создание модели {provider}: {model_config['model_name']}")
        
        if provider == "ollama":
            return ChatOllama(
                model=model_config["model_name"],
                base_url=model_config["base_url"],
                temperature=model_config["temperature"]
            )
        
        elif provider in ["openrouter", "openai"]:
            api_key = os.getenv(model_config["api_key_env"])
            return ChatOpenAI(
                model=model_config["model_name"],
                api_key=api_key,
                base_url=model_config.get("base_url"),
                temperature=model_config["temperature"]
            )
        elif provider == "deepseek":
            api_key = os.getenv(model_config["api_key_env"])
            return ChatDeepSeek(
                model=model_config["model_name"],
                api_key=api_key,
                temperature=model_config["temperature"]
            )
        else:
            raise ValueError(f"Неизвестный провайдер: {provider}")


# ===== ОСНОВНОЙ КЛАСС АГЕНТА =====
class FileSystemAgent:
    """
    Упрощенный AI-агент для работы с файловой системой
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent = None
        self.checkpointer = None
        self.mcp_client = None
        self.tools = []
        self._initialized = False
        
        logger.info(f"Создан агент с провайдером: {config.model_provider.value}")
    
    @property
    def is_ready(self) -> bool:
        """Проверяет готовность агента"""
        return self._initialized and self.agent is not None
    
    async def initialize(self) -> bool:
        """Инициализация агента"""
        if self._initialized:
            logger.warning("Агент уже инициализирован")
            return True
        
        logger.info("Инициализация агента...")
        
        try:
            # Валидация конфигурации
            self.config.validate()
            
            # Инициализация MCP клиента
            await self._init_mcp_client()
            
            # Создание модели
            model = ModelFactory.create_model(self.config)
            
            # Создание checkpointer для памяти
            if self.config.use_memory:
                self.checkpointer = InMemorySaver()
                logger.info("Память агента включена")
            
            # Создание агента
            self.agent = create_react_agent(
                model=model,
                tools=self.tools,
                checkpointer=self.checkpointer,
                prompt=self._get_system_prompt()
            )
            
            self._initialized = True
            logger.info("✅ Агент успешно инициализирован")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    @retry_on_failure()
    async def _init_mcp_client(self):
        """Инициализация MCP клиента"""
        logger.info("Инициализация MCP клиента...")
        self.mcp_client = MultiServerMCPClient(self.config.get_mcp_config())
        self.tools = await self.mcp_client.get_tools()
        
        if not self.tools:
            raise Exception("Нет доступных MCP инструментов")
        
        logger.info(f"Загружено {len(self.tools)} инструментов")
        for tool in self.tools:
            logger.info(f"  • {tool.name}")
    
    def _get_system_prompt(self) -> str:
        """Системный промпт"""
        return (
            "Ты полезный AI-ассистент для работы с файловой системой. "
            "Выполняй запросы пользователя точно и эффективно. "
            "При создании файлов/папок подтверждай успешное выполнение. "
            "Предоставляй подробную информацию о действиях. "
            "При ошибках объясняй причину и предлагай решения."
        )
    
    @retry_on_failure()
    async def process_message(self, user_input: str, thread_id: str = "default") -> str:
        """Обработка сообщения пользователя"""
        if not self.is_ready:
            return "❌ Агент не готов. Попробуйте переинициализировать."
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            message_input = {"messages": [HumanMessage(content=user_input)]}
            
            response = await self.agent.ainvoke(message_input, config)
            return response["messages"][-1].content
            
        except Exception as e:
            error_msg = f"❌ Ошибка обработки: {e}"
            logger.error(error_msg)
            return error_msg
    
    def get_status(self) -> Dict[str, Any]:
        """Информация о состоянии агента"""
        return {
            "initialized": self._initialized,
            "model_provider": self.config.model_provider.value,
            "filesystem_path": self.config.filesystem_path,
            "memory_enabled": self.config.use_memory,
            "tools_count": len(self.tools)
        }


# ===== УПРОЩЕННЫЙ ЧАТ =====
class InteractiveChat:
    """Упрощенный интерактивный чат"""
    
    def __init__(self, agent: FileSystemAgent):
        self.agent = agent
        self.thread_id = "main"
    
    def get_user_input(self) -> Optional[str]:
        """Получение ввода пользователя"""
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'выход']:
                return None
            elif user_input.lower() == 'clear':
                self.thread_id = f"thread_{asyncio.get_event_loop().time()}"
                print("💭 История очищена")
                return ""
            elif user_input.lower() == 'status':
                status = self.agent.get_status()
                print(f"📊 Статус: {status}")
                return ""
            elif not user_input:
                return ""
                
            return user_input
            
        except (KeyboardInterrupt, EOFError):
            return None
    
    async def run(self):
        """Запуск чата"""
        print("🤖 AI-агент готов к работе! (quit для выхода, status для статуса, clear для очистки)")
        
        while True:
            user_input = self.get_user_input()
            
            if user_input is None:
                break
            elif user_input == "":
                continue
            
            response = await self.agent.process_message(user_input, self.thread_id)
            print(f"\n{response}")
        
        print("\n👋 До свидания!")


# ===== ГЛАВНАЯ ФУНКЦИЯ =====
async def main():
    """Главная функция"""
    load_dotenv()
    
    try:
        # Создание конфигурации
        config = AgentConfig(
            filesystem_path=os.getenv("FILESYSTEM_PATH", "/home/yakvenalex/PythonProjects/ai_agent_project"),
            model_provider=ModelProvider(os.getenv("MODEL_PROVIDER", "deepseek"))
        )
        
        # Создание и инициализация агента
        agent = FileSystemAgent(config)
        
        if not await agent.initialize():
            logger.error("❌ Не удалось инициализировать агента")
            return
        
        # Запуск чата
        chat = InteractiveChat(agent)
        await chat.run()
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    
    logger.info("🏁 Завершение работы")


if __name__ == "__main__":
    asyncio.run(main())
