from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools.think_tool import think_tool
from tools.search_tool import esegui_ricerca_web
from tools.rag_tool import cerca_ricetta_nel_db
from tools.kg_tool import (
    controlla_storico_post,
    get_ricetta_dal_grafo,
    get_ultimi_post,
    get_ingredienti,
    get_claim_pertinenti,
    get_claim_per_retrieval,
)

load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)

GIORNI_TRA_POST = 3  # cadenza editoriale: un post ogni N giorni


# ---  BINDING DEGLI STRUMENTI (Pattern MCP) ---

lista_tools = [
    get_ultimi_post,
    esegui_ricerca_web,
    cerca_ricetta_nel_db,
    controlla_storico_post,
    get_ingredienti,
    think_tool,
    get_claim_pertinenti,
    get_ricetta_dal_grafo,
    get_claim_per_retrieval,
]


llm_con_tools = llm.bind_tools(lista_tools, parallel_tool_calls=False)
