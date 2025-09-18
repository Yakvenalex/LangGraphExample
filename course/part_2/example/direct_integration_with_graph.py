import aiohttp
import asyncio


async def ask_deepseek(api_key: str, question: str):
    """Простая функция для запроса к DeepSeek API"""
    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": question}],
        "stream": False
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            return result["choices"][0]["message"]["content"]


# Использование в узле графа
async def deepseek_node(state):
    response = await ask_deepseek(
        api_key="your_api_key",
        question=state["user_message"]
    )
    return {"ai_response": response}
