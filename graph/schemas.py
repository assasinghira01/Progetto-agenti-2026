from pydantic import BaseModel, Field, field_validator
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


class RankedDocument(BaseModel):
    id: int
    score: int = Field(description="0 irrilevante, 1 utile, 2 fondamentale")
    motivo: str = Field(
        description="Breve motivazione per lo score assegnato a questo specifico documento"
    )


class ValidationResult(BaseModel):
    is_valid: bool = Field(
        description="True se i dati approvati (score >= 1) sono sufficienti per scrivere l'articolo"
    )
    ranking_db: list[RankedDocument]
    ranking_web: list[RankedDocument]
    motivazione_qualita: str = Field(
        description="Giudizio generale finale sul set di documenti recuperato"
    )


class Ingrediente(BaseModel):
    nome: str = Field(
        description="Nome dell'ingrediente con la PRIMA LETTERA MAIUSCOLA. "
        "REGOLE DI NORMALIZZAZIONE: "
        "1. PLURALE per elementi contabili (es. 'Carota' -> 'Carote', 'Pisello' -> 'Piselli', 'Uovo' -> 'Uova'). "
        "2. SINGOLARE per masse, liquidi o elementi non contabili (es. 'Sedano', 'Latte', 'Farina', 'Burro'). "
        "3. RIMUOVI frasi superflue, imballaggi o stati fisici ovvi (es. 'Cacao amaro in polvere' -> 'Cacao amaro', 'Pomodori in scatola' -> 'Pomodori', 'Ragù fatto in casa' -> 'Ragù'). "
        "4. MANTIENI le specificità fondamentali della ricetta (es. 'Ragù alla bolognese', 'Cioccolato fondente', 'Farina 00')."
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

    # Questo validatore forza via codice la prima lettera maiuscola
    @field_validator("nome")
    @classmethod
    def capitalizza_nome(cls, v: str) -> str:
        if v:
            # Capitalizza solo la prima lettera lasciando intatto il resto (es. "Ragù alla bolognese")
            return v[0].upper() + v[1:]
        return v


class SottoRicetta(BaseModel):
    nome_specifico: str = Field(
        description="Il nome esatto usato nella ricetta, con la PRIMA LETTERA MAIUSCOLA (es. 'Ragù alla bolognese', 'La nostra besciamella')."
    )
    classe_astratta: str = Field(
        description="La radice ontologica pulita, con la PRIMA LETTERA MAIUSCOLA (es. 'Besciamella', 'Ragù'). "
        "ECCEZIONE TASSATIVA SULLE VARIANTI: Mantieni la specifica se cambia la struttura (es. 'Besciamella senza burro', 'Ragù di pesce')."
    )
    ingredienti: list[Ingrediente] = Field(
        description="Tutti e soli gli ingredienti necessari per questa specifica preparazione."
    )

    @field_validator("nome_specifico", "classe_astratta")
    @classmethod
    def capitalizza_sottoricetta(cls, v: str) -> str:
        if v:
            return v[0].upper() + v[1:]
        return v


class RecipeDraft(BaseModel):
    titolo: str = Field(
        description="Titolo accattivante per il post, con la PRIMA LETTERA MAIUSCOLA (es. 'Lasagne alla bolognese')."
    )
    introduzione: str = Field(
        description="Breve introduzione discorsiva (max 30 parole)"
    )
    sotto_ricette: list[SottoRicetta] = Field(
        description="Eventuali preparazioni secondarie che richiedono trasformazione o cottura (es. Ragù, Besciamella, Creme). "
        "NON creare sottoricette per condimenti a crudo o prodotti industriali pronti."
    )
    ingredienti_diretti: list[Ingrediente] = Field(
        description="Ingredienti principali che compongono DIRETTAMENTE la Ricetta Madre (escludendo quelli che appartengono alle sottoricette)."
    )
    preparazione: list[str] = Field(
        description="Elenco esteso e dettagliato di tutti i passaggi della ricetta. Ogni passaggio logico deve essere una stringa separata nella lista. È severamente vietato riassumere."
    )

    # [MODIFICA QUI] Da stringa singola a lista di stringhe
    fonti: list[str] = Field(
        description="Elenco di TUTTI gli URL o i percorsi dei file usati per scrivere questo post (Ricetta Madre e tutte le Sottoricette). Inserisci ogni fonte in una stringa separata della lista."
    )

    @field_validator("titolo")
    @classmethod
    def capitalizza_titolo(cls, v: str) -> str:
        if v:
            return v[0].upper() + v[1:]
        return v
