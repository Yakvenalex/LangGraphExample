from typing import TypedDict
from datetime import date
from langgraph.graph import StateGraph, START, END

from course.part_1.utils import gen_png_graph


class UserState(TypedDict):
    name: str
    surname: str
    age: int
    birth_date: date


def calculate_age(state: UserState) -> dict:
    """
    Вычисляет точный возраст человека от сегодняшней даты.
    """
    today = date.today()
    # Вычисляем разность в годах
    age = today.year - state["birth_date"].year
    # Проверяем, прошел ли уже день рождения в этом году
    if (today.month, today.day) < (state["birth_date"].month, state["birth_date"].day):
        age -= 1
    return {"age": age}


graph = StateGraph(UserState)

graph.add_node("calculate_age", calculate_age)

graph.add_edge(START, "calculate_age")
graph.add_edge("calculate_age", END)

app = graph.compile()
gen_png_graph(app, name_photo = "graph_example_1.png")
result = app.invoke({"name": "Алексей",
                     "surname": "Яковенко",
                     "birth_date": date.fromisoformat("1993-02-19")})

print(result['age'])