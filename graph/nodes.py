from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from config import llm, llm_con_tools
from graph.state import Blog_Cucina
from knowledge_graph.neo4j_manager import kg_client


class ValidationResult(BaseModel):
    is_valid: bool = Field(
        description="True se le fonti sono coerenti e contengono una ricetta logica e fattibile, False se la richiesta contiene assurdità o mancano dati fondamentali."
    )
    reasoning: str = Field(
        description="Spiegazione dettagliata del perché hai accettato o rifiutato le fonti."
    )


# Schema di output
class TopicExtraction(BaseModel):
    topic: str = Field(
        description="Il nome del piatto o ingrediente principale richiesto."
    )


# Prepariamo l'LLM strutturato da usare dentro il planner_node

llm_structured = llm.with_structured_output(TopicExtraction)


def planner_node(state: Blog_Cucina):
    print("\n--- [NODO 1: PLANNER (LLM)] ---")
    input_utente = state["input_utente"]

    # 1. Estraiamo il topic principale
    risultato = llm_structured.invoke(
        [HumanMessage(content=f"Estrai il topic: {input_utente}")]
    )
    topic_estratto = risultato.topic.capitalize()

    print(f" Topic identificato: {topic_estratto}")

    # 2. Scriviamo il topic nello stato e passiamo la palla al nodo di ricerca
    return {"topic_corrente": topic_estratto}


def krag_research_node(state: Blog_Cucina):
    print("\n--- [NODO 2: RICERCA MCP (Agente Autonomo)] ---")
    topic = state["topic_corrente"]
    messaggi = state.get("messages", [])

    istruzioni_mcp = SystemMessage(content=f"""
    Sei l'agente investigatore del blog di cucina, nel nsotro blog parleremo di ricette, sagre eventi ecc. Il topic ATTUALE su cui devi lavorare è: '{topic}'.
    
    REGOLE DI COMPORTAMENTO (rispettale rigorosamente in ordine):
    1. **Prima azione assoluta verifica che il post non sia gia stato pubblicato**: chiama 'controlla_storico_post' su '{topic}'.
    
    2. Analizza la risposta del Knowledge Graph:
       - Se inizia con "BLOCCATO": significa che il post è già stato pubblicato di recente.
       
         → Chiama 'krag_suggerisci_varianti'  se parliamo di ricette per ottenere gli ingredienti base e 3 varianti creative.
         → Dopo aver ricevuto le varianti, chiama IMMEDIATAMENTE 'chiedi_variante' passando le 3 opzioni all'utente.
         → FERMATI. Non invocare altri tool.
        
       
       - Se inizia con "OK": significa che il post è nuovo. Devi raccogliere le informazioni seguendo questo esatto flusso di ragionamento (Thought -> Action):
         
            Sai che il nostro database locale contiene solo ricette quindi non troverai nulla su sagre, eventi ecc. Effetua la ricerca su DB LOCALE usando 'cerca_ricetta_nel_db' solo se parliamo di ricette. Nel caso in cui
            L'utente richieda varianti, modifiche o versioni light/leggere delle ricette devi cercare informazioni sul web utilizzando "esegui_ricerca_web" DEVI estrarre gli INGREDIENTI, le DOSI e il PROCEDIMENTO.
            
           Nel caso in cui l'utente non parli di ricette non chiamare 'cerca_ricetta_nel_db' e vai direttamente con 'esegui_ricerca_web' per raccogliere informazioni di contesto sul '{topic}' in questione.
         
         
    
    3. Se ricevi "ERRORE" o "KRAG_ERRORE", segnala immediatamente il problema all'utente e chiedi istruzioni.
    """)

    if not messaggi:
        messaggi_da_inviare = [
            istruzioni_mcp,
            HumanMessage(content=f"Inizia la ricerca per il topic: {topic}"),
        ]
    else:
        messaggi_da_inviare = [istruzioni_mcp] + messaggi

    risposta_llm = llm_con_tools.invoke(messaggi_da_inviare)

    return {"messages": [risposta_llm]}


def validator_node(state: Blog_Cucina):
    print("\n--- [NODO 3: VALIDATORE (Fact-Checking Incrociato)] ---")
    topic = state["topic_corrente"]

    dati_db_locale = state.get("rag_documents", [])
    dati_web = state.get("web_documents", [])

    testo_db = (
        "\n".join(dati_db_locale)
        if dati_db_locale
        else "NESSUNA RICETTA TROVATA NEL DB LOCALE"
    )
    testo_web = "\n".join(dati_web) if dati_web else "NESSUN DATO COLLATERALE DAL WEB"

    prompt = f"""Analizza la fattibilità editoriale per il piatto: '{topic}'.
    
    DATI RACCOLTI DAGLI STRUMENTI (DB Locale e Web):
     
  
    === FONTE DI VERITÀ INTERNA (DB LOCALE) ===
    {testo_db}
    
    === INFORMAZIONI DI CONTESTO (RICERCA WEB) ===
    {testo_web}
    
  
    COMPITO E CRITERI DI VALUTAZIONE:
    1. **Senso Gastronomico (Coerenza)**: La richiesta ha senso logico dal punto di vista culinario? Blocca immediatamente ricette assurde, accostamenti improponibili o disgustosi (es. "Tiramisù al merluzzo", "Carbonara con la Nutella").
    2. **Sufficienza dei Dati**: Verifica semplicemente se, unendo il DB Locale e la Ricerca Web, abbiamo abbastanza informazioni (ingredienti, dosi o procedimenti minimi) per poter scrivere un articolo sensato su questo piatto.
    3. **Esito**: 
       - Imposta la validazione su True se il piatto è gastronomicamente valido e ci sono dati sufficienti per parlarne.
       - Imposta la validazione su False se il piatto è un'assurdità culinaria o se non è stato trovato assolutamente nulla in nessuna delle due fonti.
       
    """

    llm_validator = llm.with_structured_output(ValidationResult)
    esito = llm_validator.invoke([HumanMessage(content=prompt)])

    print(f" Esito Validazione: {esito.is_valid}")
    print(f" Motivazione dell'LLM: {esito.reasoning}")
    return {"is_valid": esito.is_valid}


def writer_node(state: Blog_Cucina):
    print("\n--- [NODO 4: WRITER (Sintesi e Grounding)] ---")
    topic = state["topic_corrente"]

    dati_db_locale = state.get("rag_documents", [])
    dati_web = state.get("web_documents", [])

    ha_db_locale = len(dati_db_locale) > 0
    testo_db = "\n".join(dati_db_locale) if ha_db_locale else "NESSUN DATO IN LOCALE"
    testo_web = "\n".join(dati_web) if dati_web else "NESSUN DATO DAL WEB"

    feedback = state.get("human_feedback")
    istruzione_correzione = (
        f"\nATTENZIONE - RICHIESTA DEL CAPO REDATTORE: {feedback}\nAdatta la ricetta seguendo questa istruzione."
        if feedback
        else ""
    )

    if ha_db_locale:
        regola_gerarchia = (
            "Se nei DATI_DB_LOCALE c'è la ricetta esatta (NON varianti), usa ESCLUSIVAMENTE quella.\n"
            "Se l'utente ha chiesto una variante (es. light, senza lattosio, ecc.) e il DB locale non la contiene, "
            "ignora il DB locale e usa ESCLUSIVAMENTE i DATI_RICERCA_WEB."
        )
    else:
        regola_gerarchia = (
            "Il DB Locale è vuoto. Usa ESCLUSIVAMENTE i dati della DATI_RICERCA_WEB."
        )

    # 2. Prompt di Sistema: Qui diamo SOLO le regole operative e di formattazione
    prompt_sistema = f"""Sei un food blogger professionista. Scrivi un post accattivante su: {topic}.

=== REGOLE DI STRUTTURA DEL POST ===
Il tuo output deve seguire RIGIDAMENTE questa struttura, senza eccezioni:
1. TITOLO: Un titolo accattivante per il blog.
2. INTRODUZIONE: Una breve introduzione (MAX 30 parole).
3. INGREDIENTI: Un elenco puntato con TUTTI gli ingredienti e le DOSI ESATTE (es. 500ml di latte, 50g di farina) estratti dalla fonte corretta. Non dimenticare nessuna dose!
4. PREPARAZIONE: Un riassunto breve in formato testuale narrativo (MAX 100 parole). NON USARE elenchi puntati o numerati in questa sezione, ma descrivi i passaggi in modo fluido.
5. FONTE: Cita la fonte con il link alla fine.

=== GERARCHIA DELLE FONTI (RISPETTA RIGIDAMENTE) ===
{regola_gerarchia}
{istruzione_correzione}
"""

    # 3. Contenuto del Messaggio Umano: Iniettiamo i dati puliti separati dalle regole
    contenuto_utente = f"""Ecco i dati a tua disposizione. Identifica la fonte corretta secondo le regole e genera il post.

=== DATI_DB_LOCALE ===
{testo_db}

=== DATI_RICERCA_WEB ===
{testo_web}
"""

    # Inviamo la richiesta strutturata all'LLM
    risposta_llm = llm.invoke(
        [SystemMessage(content=prompt_sistema), HumanMessage(content=contenuto_utente)]
    )

    return {"post_draft": risposta_llm.content}


def human_review_node(state: Blog_Cucina):
    print("\n--- [NODO 5: HUMAN-IN-THE-LOOP (Approvazione)] ---")
    bozza = state.get("post_draft", "")

    print("\n================ BOZZA DEL POST ================\n")
    print(bozza)
    print("\n================================================\n")

    feedback = interrupt(
        "Bozza pronta! Digita 'Approvo' per pubblicare, o scrivi le modifiche (es. 'Mettici meno sale')."
    )

    print(f" Hai risposto: {feedback}")
    # Salviamo la tua risposta nello stato
    return {"human_feedback": feedback}


class EstrattoreIngredienti(BaseModel):
    ingredienti: list[str] = Field(
        description="Lista dei soli nomi degli ingredienti principali estratti dal testo, in formato singolare e senza dosi (es. ['Melanzana', 'Pomodoro', 'Peperone'])"
    )


# Prepariamo l'LLM strutturato
llm_estrattore = llm.with_structured_output(EstrattoreIngredienti)


def kg_update_node(state: Blog_Cucina):
    print("\n--- [NODO 6: KG UPDATE (Aggiornamento Memoria)] ---")
    topic_finale = state["topic_corrente"]
    bozza_articolo = state.get("post_draft", "")

    # 1. Prompt corazzato con regole gastronomiche per estrarre la vera radice
    prompt_estrazione_radice = f"""
    Analizza l'input originario dell'utente: '{state['input_utente']}' 
    e identifica il nome della ricetta base/madre di riferimento.
    
    REGOLE TASSONOMICHE RIGIDE:
    Considera come VARIANTI solo le specifiche modifiche dietetiche, salutistiche, svuotafrigo o rivisitazioni bizzarre (es. 'Light', 'Vegana', 'Senza glutine', 'Al forno' se applicato a piatti espressi).
    Mentre considera come RADICE MADRE il nome del piatto originale senza queste specifiche modifiche. Se l'input è già molto generico e non contiene modifiche evidenti, la radice madre sarà uguale all'input stesso.
    
    Esempi di conversione:
    - 'Pasta alla carbonara light' -> Radice: 'Pasta alla carbonara' (o 'Carbonara')
    - tiramisù senza mascarpone -> Radice: 'Tiramisù'
    - 'Caponata di pesce spada' -> Radice: 'Caponata'
    - Se l'input è per esempio 'Caponata' senza specifiche aggiuntive, la radice madre è 'Caponata'.
    """

    risultato_originale = llm_structured.invoke(
        [HumanMessage(content=prompt_estrazione_radice)]
    )
    topic_originale = risultato_originale.topic.capitalize()

    # Log di controllo per verificare l'estrazione sul terminale
    print(
        f" -> [DEBUG GEOMETRIA GRAFO]: Radice Madre: '{topic_originale}' | Output Finale: '{topic_finale}'"
    )

    # 2. Estrazione degli ingredienti dal post (Come abbiamo impostato prima)
    lista_ingredienti = []
    if bozza_articolo:
        try:
            prompt_parsing = f"""Analizza la sezione 'INGREDIENTI' di questa bozza. 
            Estranni esclusivamente i nomi puliti degli ingredienti, al singolare e senza dosi.
            BOZZA:\n{bozza_articolo}"""
            risultato_ing = llm_estrattore.invoke(
                [HumanMessage(content=prompt_parsing)]
            )
            lista_ingredienti = [
                ing.strip().capitalize()
                for ing in risultato_ing.ingredienti
                if ing.strip()
            ]
        except Exception as e_parsing:
            print(f" Errore parsing ingredienti: {e_parsing}")

    # 3. Salvataggio su Neo4j
    try:
        kg_client.salva_post(topic_originale, topic_finale, lista_ingredienti)
        print(f"STORICO AGGIORNATO: Salvato su Neo4j!")
    except Exception as e:
        print(f"Errore su Neo4j: {str(e)}")

    return {}


def aggiorna_topic_node(state: Blog_Cucina):
    """
    Usa un mini-LLM strutturato per estrarre in modo sicuro la variante scelta
    evitando la fragilità dello split testuale manuale.
    """
    print("\n--- [NODO: AGGIORNA TOPIC] ---")

    # 1. Recuperiamo il testo digitato dall'utente.
    # Se arriva dal tool chiedi_variante (interrupt nel ciclo di ricerca), lo troviamo nell'ultimo messaggio.
    # Se arriva dalla revisione finale, lo troviamo in human_feedback.
    ultimo_messaggio = state["messages"][-1] if state.get("messages") else None

    scelta_utente = state.get("human_feedback")

    if not scelta_utente and ultimo_messaggio:
        scelta_utente = ultimo_messaggio.content

    if not scelta_utente:
        print(" [AGGIORNA TOPIC] Nessun input utente trovato. Salto.")
        return {}

    # 2. Parsing sicuro con l'LLM Strutturato
    prompt_parsing = f"""L'utente ha selezionato una variante tra quelle proposte o ha indicato una nuova rotta. 
    Estrarre SOLO il nome pulito del piatto o della variante scelto da questo testo: '{scelta_utente}'"""

    risultato = llm_structured.invoke([HumanMessage(content=prompt_parsing)])
    nuovo_topic = risultato.topic.capitalize()

    print(
        f" [AGGIORNA TOPIC] Rilevata scelta utente. Nuovo topic impostato a: '{nuovo_topic}'"
    )
    return {"topic_corrente": nuovo_topic}
