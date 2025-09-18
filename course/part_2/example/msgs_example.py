from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()


llm = ChatDeepSeek(model="deepseek-chat")

messages = [
    SystemMessage(content="Ты полезный программист-консультант"),
    HumanMessage(content="Как написать цикл в Python?"),
    AIMessage(content="Используйте for или while. Пример: for i in range(10):"),
    HumanMessage(content="А что такое range?")
]

# Отправляем структурированную историю диалога
response = llm.invoke(messages)
print(response.content)
