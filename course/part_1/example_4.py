from typing import TypedDict, Optional
from datetime import date, timedelta
from langgraph.graph import StateGraph, START, END

from course.part_1.utils import gen_png_graph


class UserState(TypedDict):
    age: int
    message: str


def fake_node(state: UserState) -> UserState:
    return state


def check_age(state: UserState) -> str:
    return "совершеннолетний" if state["age"] >= 18 else "не совершеннолетний"


def generate_success_message(state: UserState) -> dict:
    return {"message": f"Вам уже {state['age']} и вы можете водить!"}


def generate_failure_message(state: UserState) -> dict:
    return {"message": f"Вам еще только {state['age']} и вы не можете водить."}


graph = StateGraph(UserState)

graph.add_node("fake_node", lambda state: state)
graph.add_node("generate_success_message", generate_success_message)
graph.add_node("generate_failure_message", generate_failure_message)

graph.add_edge(START, "fake_node")
graph.add_conditional_edges(
    "fake_node",
    check_age,
    {
        "совершеннолетний": "generate_success_message",
        "не совершеннолетний": "generate_failure_message"
    }
)

graph.add_edge("generate_success_message", END)
graph.add_edge("generate_failure_message", END)

app = graph.compile()
gen_png_graph(app, name_photo = "graph_example_4.png")
result = app.invoke({"age": 17})

print(result)
