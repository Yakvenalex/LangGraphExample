from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
load_dotenv()

llm = GigaChat(model="GigaChat-2-Max", verify_ssl_certs=False)

response = llm.invoke("Кто тебя создал?")
print(response)
