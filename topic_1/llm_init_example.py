import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek

load_dotenv()


def get_openai_llm():
    return ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def get_deepseek_llm():
    return ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))


def get_ollama_llm(model="qwen2.5:32b", base_url="http://localhost:11434"):
    return ChatOllama(
        model=model,
        temperature=0,
        base_url=base_url
    )


def get_openrouter_llm(model="moonshotai/kimi-k2:free"):
    return ChatOpenAI(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )


if __name__ == "__main__":
    llm = get_openrouter_llm()
    response = llm.invoke("Кто ты?")
    print(response.content)
