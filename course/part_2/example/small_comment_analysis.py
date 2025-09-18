from langchain_amvera import AmveraLLM
from pydantic import BaseModel, Field
from typing import List, Literal
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ Определяем структуру данных
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Тональность отзыва: положительная, отрицательная или нейтральная"
    )
    confidence: float = Field(
        description="Уверенность в анализе от 0.0 до 1.0",
        ge=0.0, le=1.0
    )
    key_topics: List[str] = Field(
        description="Ключевые темы, упомянутые в отзыве",
        max_items=5
    )
    summary: str = Field(
        description="Краткое резюме отзыва в одном предложении",
        max_length=200
    )


# 2️⃣ Создаем парсер
parser = JsonOutputParser(pydantic_object=SentimentAnalysis)

# 3️⃣ Создаем умный шаблон
prompt_template = PromptTemplate(
    template="""Проанализируй отзыв: {review}

{format_instructions}

ТОЛЬКО JSON!""",
    input_variables=["review"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()  # Автомагия!
    }
)

# 4️⃣ Инициализируем нейросеть
llm = AmveraLLM(model="llama70b", temperature=0.0)

# Тестовый отзыв
review = "Товар отличный, быстрая доставка! Очень доволен покупкой."

print("=== ПОШАГОВОЕ ВЫПОЛНЕНИЕ ===")

# Шаг 1: Применяем шаблон
print("1️⃣ Применяем PromptTemplate")
prompt_value = prompt_template.invoke({"review": review})
print(f"Тип: {type(prompt_value)}")

# Посмотрим на готовый промпт
prompt_text = prompt_value.to_string()
print("Готовый промпт:")
print(prompt_text[:200] + "...")  # Первые 200 символов
print()

# Шаг 2: Отправляем в нейросеть
print("2️⃣ Отправляем в нейросеть")
llm_response = llm.invoke(prompt_value)
print(f"Тип ответа: {type(llm_response)}")
print(f"Ответ: {llm_response.content}")
print()

# Шаг 3: Парсим JSON
print("3️⃣ Парсим JSON")
parsed_result = parser.invoke(llm_response)
print(f"Тип результата: {type(parsed_result)}")
print("Структурированные данные:")
for key, value in parsed_result.items():
    print(f"  {key}: {value}")
