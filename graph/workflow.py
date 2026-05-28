from langgraph.graph import StateGraph, START, END
from graph.state import Blog_Cucina
from graph.nodes import planner_node, krag_research_node, writer_node, validator_node

def check_validation(state: Blog_Cucina):
    """Legge lo stato del validatore e decide dove indirizzare il flusso."""
    if state.get("is_valid") == True:
        print("ROUTING: Validazione superata! Procedo alla scrittura.")
        return "writer"
    else:
        print("ROUTING: Rilevata assurdità o mancanza di dati. Blocco il processo!")
        return "end_block"

builder = StateGraph(Blog_Cucina)

# Registrazione dei nodi
builder.add_node("planner", planner_node)
builder.add_node("research", krag_research_node)
builder.add_node("validator", validator_node) 
builder.add_node("writer", writer_node) 

# Definizione degli archi standard
builder.add_edge(START, "planner")
builder.add_edge("planner", "research")
builder.add_edge("research", "validator")

# Definizione dell'arco condizionale dal validatore
builder.add_conditional_edges(
    "validator",
    check_validation,
    {
        "writer": "writer",
        "end_block": END  # Se è falso, interrompe l'esecuzione evitando allucinazioni
    }
)

builder.add_edge("writer", END)

app = builder.compile()