from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage
from graph.state import Blog_Cucina
from graph.nodes import (
    planner_node,
    krag_research_node,
    human_variant_node,
    writer_node,
    validator_node,
    human_review_node,
    kg_update_node,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from config import lista_tools  #

# --- 1. IL NODO DEGLI STRUMENTI ---
# LangGraph crea automaticamente il nodo che esegue le funzioni Python
nodo_strumenti = ToolNode(lista_tools)


# --- 2. FUNZIONI DI ROUTING ---
def router_ricerca(state: Blog_Cucina):
    """Controlla se l'agente ha deciso di usare un tool o ha finito."""
    ultimo_messaggio = state["messages"][-1]

    # Se l'LLM ha richiesto l'uso di uno strumento (es. ha chiamato cerca_ricetta_nel_db)
    if hasattr(ultimo_messaggio, "tool_calls") and ultimo_messaggio.tool_calls:
        print(
            f"  L'Agente sta usando i tool: {[t['name'] for t in ultimo_messaggio.tool_calls]}"
        )
        return "tools"

    if (
        "?" in ultimo_messaggio.content
        and "variante" in ultimo_messaggio.content.lower()
    ):
        return "human_variant"

    # Se non ha chiamato tool, significa che ha finito di raccogliere dati
    print(" L'Agente ha concluso la ricerca. Passo al validatore.")
    return "validator"


def check_validation(state: Blog_Cucina):
    """Legge lo stato del validatore e decide dove indirizzare il flusso."""
    if state.get("is_valid") == True:
        print("ROUTING: Validazione superata! Procedo alla scrittura.")
        return "writer"
    else:
        print("ROUTING: Rilevata assurdità o mancanza di dati. Blocco il processo!")
        return "end_block"


def check_human_approval(state: Blog_Cucina):
    """Legge la decisione umana presa durante l'interruzione."""
    feedback = state.get("human_feedback", "").lower()

    if "approvo" in feedback or "ok" in feedback or "perfetto" in feedback:
        print(" Semaforo VERDE umano: Procedo al salvataggio su Neo4j.")
        return "kg_update"
    else:
        print(" Semaforo ARANCIONE umano: L'utente vuole modifiche. Riscrivo il post!")
        return "writer"


builder = StateGraph(Blog_Cucina)

# Registrazione dei nodi
builder.add_node("planner", planner_node)
builder.add_node("research", krag_research_node)
builder.add_node("human_variant", human_variant_node)
builder.add_node("validator", validator_node)
builder.add_node("writer", writer_node)
builder.add_node("human_review", human_review_node)
builder.add_node("kg_update", kg_update_node)
builder.add_node("tools", nodo_strumenti)

# Definizione degli archi standard
builder.add_edge(START, "planner")
builder.add_edge("planner", "research")

builder.add_conditional_edges(
    "research",
    router_ricerca,
    {"tools": "tools", "validator": "validator", "human_variant": "human_variant"},
)


builder.add_edge("tools", "research")
builder.add_edge("human_variant", "research")

# Definizione dell'arco condizionale dal validatore
builder.add_conditional_edges(
    "validator",
    check_validation,
    {
        "writer": "writer",
        "end_block": END,  # Se è falso, interrompe l'esecuzione evitando allucinazioni
    },
)

# Dal Writer andiamo alla Revisione Umana
builder.add_edge("writer", "human_review")

# L'arco condizionale del feedback umano (Loop di Correzione)
builder.add_conditional_edges(
    "human_review", check_human_approval, {"kg_update": "kg_update", "writer": "writer"}
)

# Dal salvataggio andiamo alla fine
builder.add_edge("kg_update", END)
memoria_temporanea = InMemorySaver()
app = builder.compile(checkpointer=memoria_temporanea)
