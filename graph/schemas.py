from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class TopicPianificato(BaseModel):
    topic: str = Field(
        description="Il nome specifico dell'argomento (es. 'Cassata Siciliana', 'Pasta allo scoglio')"
    )

    categoria: Literal[
        "Antipasto", "Primo", "Secondo", "Contorno", "Dolce", "Salse e Condimenti"
    ] = Field(description="La categoria gastronomica del piatto.")

    giustificazione: Optional[str] = Field(
        default="",
        description="Se è una variante o fa parte di un piano editoriale, spiega il motivo della scelta. Se è un topic singolo imposto dall'utente, lascia vuoto.",
    )


class PianoEditoriale(BaseModel):
    sequenza_post: List[TopicPianificato] = Field(
        description="La lista ordinata di topic pianificati."
    )


class ValidationResult(BaseModel):
    is_valid: bool = Field(
        description=(
            "True se le fonti sono coerenti e contengono una ricetta "
            "logica e fattibile. False se la richiesta contiene assurdità "
            "o mancano dati fondamentali."
        )
    )

    reasoning: str = Field(description="Spiegazione della validazione generale.")

    usa_db_locale: bool = Field(
        description=(
            "True se i documenti recuperati dal DB locale sono pertinenti "
            "al topic richiesto. False se il retrieval locale ha restituito "
            "ricette completamente diverse o non utili."
        )
    )

    documenti_web_approvati: List[int] = Field(
        description=(
            "Lista degli ID numerici dei documenti web approvati. "
            "Usa gli ID mostrati come WEB_DOC_X. "
            "Normalmente deve contenere un solo ID."
        )
    )

    documenti_db_approvati: List[int] = Field(
        default_factory=list,
        description="Lista contenente gli ID dei documenti del DB locale ritenuti pertinenti.",
    )
    motivazione_qualita: str = Field(
        description=(
            "Motiva brevemente la scelta della fonte web approvata "
            "e l'eventuale esclusione delle altre."
        )
    )


class Ingrediente(BaseModel):
    nome: str = Field(
        description="Nome dell'ingrediente. REGOLA DI NORMALIZZAZIONE: Usare il singolare e rimuovere aggettivi inutili e frasi superflue. "
        "(es. 'Uova fresche' -> 'Uovo', 'Pomodorini biologici' -> 'Pomodorino', 'Besciamella classica' -> 'Besciamella', 'ragù fatto in casa' -> 'Ragù'). "
        "MANTIENI invece la tipologia fondamentale se cambia l'ingrediente (es. 'Farina 00', 'Carne macinata di maiale')."
    )
    quantita: str = Field(description="Quantità completa (es. '300g')")
    fase_utilizzo: str = Field(
        description="Categorizza l'uso logico di questo ingrediente nel piatto finito, usando SEMPRE una di queste categorie standard: "
        "'Impasto' (tutto ciò che va mescolato per formare una base solida), "
        "'Condimento' (ingredienti crudi o pronti aggiunti sopra una base o in un'insalata, es. pomodoro su pizza, olio a crudo), "
        "'Farcitura' (ingredienti inseriti all'interno di una chiusura, es. ripieno di un raviolo o di un calzone), "
        "'Panatura' (elementi per coprire l'esterno prima della frittura/cottura), "
        "'Guarnizione' (elementi finali puramente decorativi o di tocco finale, es. prezzemolo tritato alla fine), "
        "'Cottura' (elementi usati solo come mezzo tecnico, es. olio per friggere, brodo per sfumare). "
        "Se l'ingrediente costituisce la massa principale del piatto e non rientra in queste categorie, scrivi 'Base'."
    )


class SottoRicetta(BaseModel):
    nome_specifico: str = Field(
        description="Il nome esatto usato nella ricetta (es. 'Ragù della nonna', 'La nostra besciamella')."
    )
    classe_astratta: str = Field(
        description="La radice ontologica pulita della preparazione. "
        "REGOLA DI NORMALIZZAZIONE: Rimuovi aggettivi generici, di origine o di qualità (es. 'Besciamella classica' -> 'Besciamella', "
        "'Ragù bolognese' -> 'Ragù'). "
        "ECCEZIONE TASSATIVA SULLE VARIANTI: Se il nome contiene modifiche strutturali, dietetiche o ingredienti alternativi "
        "(es. 'Besciamella senza burro', 'Crema vegana', 'Ragù di pesce', 'Maionese senza uova'), "
        "la classe astratta DEVE mantenere questa specifica per intero (es. 'Besciamella senza burro'). Non semplificarla."
    )
    ingredienti: list[Ingrediente] = Field(
        description="Tutti e soli gli ingredienti necessari per questa specifica preparazione."
    )


class RecipeDraft(BaseModel):
    titolo: str = Field(description="Titolo accattivante per il post")
    introduzione: str = Field(description="Breve introduzione discorsiva")
    sotto_ricette: list[SottoRicetta] = Field(
        description="Eventuali preparazioni secondarie e indipendenticon i loro ingredienti esclusivi."
        "La fonte nella maggior parte dei casi, specificherà se ci saranno o meno delle preparazioni secondarie. "
        "Se non sono presenti, dovrai essere in grado di identificare autonomamente se è necessario creare una sotto-ricetta (es. ragù, besciamella, maionese, pastella, guarnizioni ecc...) o se tutti gli ingredienti"
        "e la preparazione possono essere inseriti direttamente nella ricetta principale."
        "REGOLA SUI PRODOTTI PRONTI: Crea una sotto-ricetta SOLO SE la fonte elenca gli ingredienti crudi per prepararla da zero (es. burro, farina e latte per la besciamella). "
        "Se la fonte utilizza un prodotto industriale già pronto (es. '500ml di besciamella pronta', '1 vasetto di pesto', 'Pasta sfoglia comprata'), "
        "NON creare la sotto-ricetta, ma inserisci il prodotto finito direttamente tra gli 'ingredienti_diretti'."
        "REGOLA SUI CONDIMENTI/TOPPING: NON creare MAI una sotto-ricetta per semplici raggruppamenti di ingredienti crudi o pronti da disporre su un piatto (es. 'Per condire', 'Topping', 'Per guarnire', 'Per la farcitura',ecc..). "
        "Ingredienti base, DEVONO andare negli 'ingredienti_diretti', anche se la fonte web li raggruppa sotto un titolo separato. "
        "Una sotto-ricetta (es. Biga, Impasto, Ragù, Crema, ecc...) implica una trasformazione o cottura congiunta degli ingredienti."
    )
    ingredienti_diretti: list[Ingrediente] = Field(
        description="Ingredienti principali che compongono DIRETTAMENTE il piatto. "
        "REGOLA TASSATIVA DI MUTUA ESCLUSIVITÀ: Gli ingredienti inseriti qui NON DEVONO comparire"
        "all'interno di nessuna 'sotto_ricetta', nel caso in cui trovi degli ingredienti duplicati inseriscili solo nel contesto corretto."
        "Le duplicazioni di ingredienti tra ingredienti_diretti e sotto_ricette sono un errore logico che indica una confusione tra preparazione principale e secondaria. "
        " Se un ingrediente appartiene al ragù, mettilo SOLO "
        "nella sotto-ricetta del ragù ed escludilo totalmente da questa lista."
    )
    preparazione: str = Field(description="Testo discorsivo del procedimento")
    fonte: str = Field(description="L'URL della fonte da cui è tratta la ricetta")
