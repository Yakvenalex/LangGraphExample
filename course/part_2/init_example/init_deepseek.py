from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


load_dotenv()


llm = ChatDeepSeek(model="deepseek-chat")

# # Способ 1: автоматическое оборачивание
# response = llm.invoke("Кто тебя создал?")
# # <class 'langchain_core.messages.ai.AIMessage'>
# print(f"Тип ответа: {type(response)}")
# print(f"Содержимое: {response.content}")

# Способ 2: явное создание HumanMessage
human_msg = HumanMessage(content="Кто тебя создал?")
response = llm.invoke([human_msg])  # Передаём список сообщений


print(f"Тип ответа: {type(response)}")
print(f"Содержимое: {response.content}")
print(response)
