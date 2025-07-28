import asyncio
import json
from typing import TypedDict, Dict, Any
from enum import Enum
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage


# Категории профессий
CATEGORIES = [
    "2D-аниматор", "3D-аниматор", "3D-моделлер", "Бизнес-аналитик", 
    "Блокчейн-разработчик", "Бухгалтер", "Бэкенд-разработчик (Node.js, Python, PHP, Ruby)",
    "Видео-продюсер", "Видеомонтажер", ...
]


class JobType(Enum):
    PROJECT = "проектная работа"
    PERMANENT = "постоянная работа"


class SearchType(Enum):
    LOOKING_FOR_WORK = "поиск работы"
    LOOKING_FOR_PERFORMER = "поиск исполнителя"


class State(TypedDict):
    """Состояние агента для хранения информации о процессе классификации"""
    description: str
    job_type: str
    category: str
    search_type: str
    confidence_scores: Dict[str, float]
    processed: bool


class VacancyClassificationAgent:
    """Асинхронный агент для классификации вакансий и услуг"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Инициализация агента"""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Создает рабочий процесс агента на основе LangGraph"""
        workflow = StateGraph(State)
        
        # Добавляем узлы в граф
        workflow.add_node("job_type_classification", self._classify_job_type)
        workflow.add_node("category_classification", self._classify_category)
        workflow.add_node("search_type_classification", self._classify_search_type)
        workflow.add_node("confidence_calculation", self._calculate_confidence)
        
        # Определяем последовательность выполнения узлов
        workflow.set_entry_point("job_type_classification")
        workflow.add_edge("job_type_classification", "category_classification")
        workflow.add_edge("category_classification", "search_type_classification")
        workflow.add_edge("search_type_classification", "confidence_calculation")
        workflow.add_edge("confidence_calculation", END)
        
        return workflow.compile()
    
    async def _classify_job_type(self, state: State) -> Dict[str, Any]:
        """Узел для определения типа работы: проектная или постоянная"""
        prompt = PromptTemplate(
            input_variables=["description"],
            template="""
            Проанализируй следующее описание и определи тип работы.
            
            Описание: {description}
            
            Ответь только одним из двух вариантов:
            - "проектная работа" - если это временная задача, проект, фриланс, разовая работа
            - "постоянная работа" - если это постоянная должность, штатная позиция, долгосрочное трудоустройство
            
            Тип работы:
            """
        )
        
        message = HumanMessage(content=prompt.format(description=state["description"]))
        response = await self.llm.ainvoke([message])
        job_type = response.content.strip().lower()
        
        # Нормализуем ответ
        if "проектная" in job_type or "проект" in job_type or "фриланс" in job_type:
            job_type = JobType.PROJECT.value
        else:
            job_type = JobType.PERMANENT.value
            
        return {"job_type": job_type}
    
    async def _classify_category(self, state: State) -> Dict[str, Any]:
        """Узел для определения категории профессии"""
        categories_str = "\n".join([f"- {cat}" for cat in CATEGORIES])
        
        prompt = PromptTemplate(
            input_variables=["description", "categories"],
            template="""
            Проанализируй описание вакансии/услуги и определи наиболее подходящую категорию из списка.
            
            Описание: {description}
            
            Доступные категории:
            {categories}
            
            Выбери ТОЧНО одну категорию из списка выше, которая лучше всего соответствует описанию.
            Ответь только названием категории без дополнительных пояснений.
            
            Категория:
            """
        )
        
        message = HumanMessage(content=prompt.format(
            description=state["description"], 
            categories=categories_str
        ))
        response = await self.llm.ainvoke([message])
        category = response.content.strip()
        
        # Проверяем, есть ли категория в списке доступных
        if category not in CATEGORIES:
            # Ищем наиболее похожую категорию
            category = self._find_closest_category(category)
            
        return {"category": category}
    
    async def _classify_search_type(self, state: State) -> Dict[str, Any]:
        """Узел для определения типа поиска"""
        prompt = PromptTemplate(
            input_variables=["description"],
            template="""
            Проанализируй описание и определи, кто и что ищет.
            
            Описание: {description}
            
            Ответь только одним из двух вариантов:
            - "поиск работы" - если соискатель ищет работу/заказы
            - "поиск исполнителя" - если работодатель/заказчик ищет исполнителя
            
            Обрати внимание на ключевые слова:
            - "ищу работу", "резюме", "хочу работать" = поиск работы
            - "требуется", "ищем", "вакансия", "нужен специалист" = поиск исполнителя
            
            Тип поиска:
            """
        )
        
        message = HumanMessage(content=prompt.format(description=state["description"]))
        response = await self.llm.ainvoke([message])
        search_type = response.content.strip().lower()
        
        # Нормализуем ответ
        if "поиск работы" in search_type or "ищу работу" in search_type:
            search_type = SearchType.LOOKING_FOR_WORK.value
        else:
            search_type = SearchType.LOOKING_FOR_PERFORMER.value
            
        return {"search_type": search_type}
    
    async def _calculate_confidence(self, state: State) -> Dict[str, Any]:
        """Узел для расчета уровня уверенности в классификации"""
        prompt = PromptTemplate(
            input_variables=["description", "job_type", "category", "search_type"],
            template="""
            Оцени уверенность классификации по шкале от 0.0 до 1.0 для каждого параметра:
            
            Описание: {description}
            Тип работы: {job_type}
            Категория: {category}
            Тип поиска: {search_type}
            
            Ответь в формате JSON:
            {{
                "job_type_confidence": 0.0-1.0,
                "category_confidence": 0.0-1.0,
                "search_type_confidence": 0.0-1.0
            }}
            """
        )
        
        message = HumanMessage(content=prompt.format(
            description=state["description"],
            job_type=state["job_type"],
            category=state["category"],
            search_type=state["search_type"]
        ))
        response = await self.llm.ainvoke([message])
        
        try:
            confidence_scores = json.loads(response.content.strip())
        except:
            # Fallback значения если парсинг не удался
            confidence_scores = {
                "job_type_confidence": 0.7,
                "category_confidence": 0.7,
                "search_type_confidence": 0.7
            }
        
        return {
            "confidence_scores": confidence_scores,
            "processed": True
        }
    
    def _find_closest_category(self, predicted_category: str) -> str:
        """Находит наиболее похожую категорию из списка доступных"""
        # Простая эвристика поиска по вхождению ключевых слов
        predicted_lower = predicted_category.lower()
        
        for category in CATEGORIES:
            category_lower = category.lower()
            if predicted_lower in category_lower or category_lower in predicted_lower:
                return category
                
        # Если ничего не найдено, возвращаем первую категорию как fallback
        return CATEGORIES[0]
    
    async def classify(self, description: str) -> Dict[str, Any]:
        """Основной метод для классификации вакансии/услуги"""
        initial_state = {
            "description": description,
            "job_type": "",
            "category": "",
            "search_type": "",
            "confidence_scores": {},
            "processed": False
        }
        
        # Запускаем рабочий процесс
        result = await self.workflow.ainvoke(initial_state)
        
        # Формируем итоговый ответ в формате JSON
        classification_result = {
            "job_type": result["job_type"],
            "category": result["category"],
            "search_type": result["search_type"],
            "confidence_scores": result["confidence_scores"],
            "success": result["processed"]
        }
        
        return classification_result


async def main():
    """Демонстрация работы агента"""
    agent = VacancyClassificationAgent()
    
    # Тестовые примеры
    test_cases = [
        "Требуется Python разработчик для создания веб-приложения на Django. Постоянная работа, полный рабочий день.",
        "Ищу заказы на создание логотипов и фирменного стиля. Работаю в Adobe Illustrator.",
        "Нужен 3D-аниматор для краткосрочного проекта создания рекламного ролика.",
        "Резюме: опытный маркетолог, ищу удаленную работу в сфере digital-маркетинга",
        "Ищем фронтенд-разработчика React в нашу команду на постоянную основе"
    ]
  
    print("🤖 Демонстрация работы агента классификации вакансий\n")
    
    for i, description in enumerate(test_cases, 1):
        print(f"📋 Тест {i}:")
        print(f"Описание: {description}")
        
        try:
            result = await agent.classify(description)
            print("Результат классификации:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            
        print("-" * 80)


if __name__ == "__main__":
    # Для запуска нужно установить переменную окружения OPENAI_API_KEY
    asyncio.run(main())
