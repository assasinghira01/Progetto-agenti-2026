import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

# Embedding locale
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Agente intelligente OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.1,
    max_tokens=600
)

# Schema di output
class TopicExtraction(BaseModel):
    topic: str = Field(description="Il nome del piatto o ingrediente principale richiesto.")

llm_structured = llm.with_structured_output(TopicExtraction)