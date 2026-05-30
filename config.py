import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from tools.search_tool import esegui_ricerca_web
from tools.rag_tool import cerca_ricetta_nel_db
from tools.kg_tool import controlla_storico_post

load_dotenv()


# Agente intelligente OpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=600)


# --- IL BINDING DEGLI STRUMENTI (Pattern MCP) ---
# Impacchettiamo i nostri 3 strumenti obbligatori
lista_tools = [esegui_ricerca_web, cerca_ricetta_nel_db, controlla_storico_post]

# 2. Creiamo un "Super-Cervello" dotato di mani.
# llm_con_tools è la variabile che useremo nel nostro Nodo di Ricerca!
llm_con_tools = llm.bind_tools(lista_tools)
