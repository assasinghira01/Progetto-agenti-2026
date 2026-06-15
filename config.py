from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools.think_tool import think_tool
from tools.search_tool import esegui_ricerca_web
from tools.rag_tool import cerca_ricetta_nel_db
from tools.kg_tool import (
    controlla_storico_post,
    get_ultimi_post,
    get_ingredienti_per_variante,
)

load_dotenv()


# Agente intelligente OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=800)


# --- IL BINDING DEGLI STRUMENTI (Pattern MCP) ---
# Impacchettiamo i nostri 3 strumenti obbligatori
lista_tools = [
    get_ultimi_post,
    esegui_ricerca_web,
    cerca_ricetta_nel_db,
    controlla_storico_post,
    get_ingredienti_per_variante,
    think_tool,
]

# 2. Creiamo un "Super-Cervello" dotato di mani.
# llm_con_tools è la variabile che useremo nel nostro Nodo di Ricerca!
llm_con_tools = llm.bind_tools(lista_tools, parallel_tool_calls=False)
