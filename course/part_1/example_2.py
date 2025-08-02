from typing import TypedDict
from datetime import date
from langgraph.graph import StateGraph, START, END

from course.part_1.utils import gen_png_graph


class UserState(TypedDict):
    name: str
    surname: str
    age: int
    birth_date: date
    message: str


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


def check_drive(state: UserState) -> str:
    if state["age"] >= 18:
        return "можно"
    else:
        return "нельзя"


def generate_success_message(state: UserState) -> dict:
    return {"message": f"Поздравляем, {state['name']} {state['surname']} вам уже {state['age']} и вы можете водить!"}


def generate_failure_message(state: UserState) -> dict:
    return {"message": f"К сожалению, {state['name']} {state['surname']} вам еще только {state['age']} и "
                       f"вы не можете водить."}


graph = StateGraph(UserState)

graph.add_node("calculate_age", calculate_age)
graph.add_node("generate_success_message", generate_success_message)
graph.add_node("generate_failure_message", generate_failure_message)

graph.add_edge(START, "calculate_age")
graph.add_conditional_edges("calculate_age",
                            check_drive,
                            {"можно": "generate_success_message",
                            "нельзя": "generate_failure_message"})

graph.add_edge("generate_success_message", END)
graph.add_edge("generate_failure_message", END)

app = graph.compile()
gen_png_graph(app, name_photo = "graph_example_2.png")
result = app.invoke({"name": "Алексей",
                     "surname": "Яковенко",
                     "birth_date": date.fromisoformat("1993-02-19")})

print(result)
