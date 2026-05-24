from langgraph.graph import StateGraph, START, END
from graph.state import Blog_Cucina
from graph.nodes import planner_node, krag_research_node, writer_node

builder = StateGraph(Blog_Cucina)

builder.add_node("nodo1", planner_node)
builder.add_node("nodo2", krag_research_node)
builder.add_node("nodo3", writer_node) 

builder.add_edge(START, "nodo1")
builder.add_edge("nodo1", "nodo2")
builder.add_edge("nodo2", "nodo3")
builder.add_edge("nodo3", END)

app = builder.compile()