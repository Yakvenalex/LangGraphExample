from langchain_amvera import AmveraLLM
from dotenv import load_dotenv

load_dotenv()

llm = AmveraLLM(model="llama70b")

response = llm.invoke("Кто тебя создал?")
print(response.content)
