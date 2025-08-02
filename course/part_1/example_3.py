from typing import TypedDict, Optional
from datetime import date, timedelta
from langgraph.graph import StateGraph, START, END

from course.part_1.utils import gen_png_graph


class UserState(TypedDict):
    name: str
    surname: str
    age: int
    birth_date: date
    today: date
    message: str


def calculate_age(state: UserState) -> dict:
    """
    Вычисляет точный возраст человека от сегодняшней даты.
    """
    today = state["today"]
    # Вычисляем разность в годах
    age = today.year - state["birth_date"].year
    # Проверяем, прошел ли уже день рождения в этом году
    if (today.month, today.day) < (state["birth_date"].month, state["birth_date"].day):
        age -= 1
    return {"age": age, "today": today}


def autoincrement_date(state: UserState) -> dict:
    new_date = state["today"] + timedelta(days=1)
    print(f"{state['today']} -> {new_date}")
    return {"today": new_date}


def check_drive(state: UserState) -> str:
    return "можно" if state["age"] >= 18 else "нельзя"


def generate_success_message(state: UserState) -> dict:
    return {"message": f"Поздравляем, {state['name']} {state['surname']}! "
                       f"Вам уже {state['age']} и вы можете водить!"}

graph = StateGraph(UserState)

graph.add_node("calculate_age", calculate_age)
graph.add_node("generate_success_message", generate_success_message)
graph.add_node("autoincrement_date", autoincrement_date)

graph.add_edge(START, "calculate_age")
graph.add_conditional_edges(
    "calculate_age",
    check_drive,
    {
        "можно": "generate_success_message",
        "нельзя": "autoincrement_date"
    }
)

graph.add_edge("generate_success_message", END)
graph.add_edge("autoincrement_date", "calculate_age")

app = graph.compile()
gen_png_graph(app, name_photo = "graph_example_3.png")
result = app.invoke({
    "name": "Алексей",
    "surname": "Яковенко",
    "birth_date": date.fromisoformat("2008-02-19"),
    "today": date.today()
},
    {"recursion_limit": 1000})

print(result)
