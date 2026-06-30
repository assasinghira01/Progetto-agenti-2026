import os
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from datetime import datetime
from langchain_core.messages import HumanMessage


class CucinaKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        print("[NEO4J] Inizializzazione embeddings locali per il Grafo...")
        # Usiamo lo STESSO modello di ChromaDB.
        # Questo modello produce vettori a 1024 dimensioni.
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # Creiamo l'indice vettoriale
        self._crea_indice_vettoriale()

    def _crea_indice_vettoriale(self):
        """Prepara il database a grafo per la ricerca semantica."""
        query = """
        CREATE VECTOR INDEX recipe_embedding_index IF NOT EXISTS
        FOR (r:Ricetta) ON (r.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 1024,
          `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)
            print(
                " [NEO4J] Indice vettoriale 'recipe_embedding_index' verificato/creato."
            )

    def init_blog_style(
        self,
        tono: str = "amichevole e appassionato",
        registro: str = "informale ma preciso, ricco di aneddoti culturali",
        lunghezza_target: int = 600,
        audience: str = "appassionati di cucina tradizionale, livello intermedio",
        note_stilistiche: str = (
            "Usa sempre la seconda persona singolare per coinvolgere il lettore. "
            "Includi almeno un rimando culturale o storico per ogni ricetta. "
            "Evita termini tecnici senza spiegarli. "
            "Il titolo deve essere evocativo, non descrittivo."
        ),
    ):
        """
        Crea (o aggiorna) il nodo singleton BlogStyle e lo collega al Blog
        tramite USA_STILE. Il campo aggiornato_il tiene traccia della versione
        dello stile: i Post collegati via SCRITTO_CON (creata in salva_post)
        restano tracciabili anche se lo stile cambia in futuro.
        """
        data_oggi = datetime.now().strftime("%Y-%m-%d")
        with self.driver.session() as session:
            session.run(
                """
                MERGE (b:Blog { name: "Il mio Blog di Cucina" })
                MERGE (s:BlogStyle { name: "stile_blog" })
                SET s.tono              = $tono,
                    s.registro          = $registro,
                    s.lunghezza_target  = $lunghezza_target,
                    s.audience          = $audience,
                    s.note_stilistiche  = $note_stilistiche,
                    s.aggiornato_il     = $data
                MERGE (b)-[:USA_STILE]->(s)
                """,
                tono=tono,
                registro=registro,
                lunghezza_target=lunghezza_target,
                audience=audience,
                note_stilistiche=note_stilistiche,
                data=data_oggi,
            )
        print("[NEO4J] Nodo BlogStyle aggiornato e collegato al Blog via USA_STILE.")

    def get_contesto_editoriale(self, topic_corrente: str, n_post: int = 3) -> dict:
        """
        Recupera dal KG il contesto editoriale completo per il writer_node:
          - Linee guida stilistiche del blog (nodo BlogStyle)
          - Claim chiave dei post recenti NON relativi al topic corrente
            (per creare rimandi o evitare ripetizioni)
          - Topic già trattati di recente (per segnalare connessioni tematiche)
        """
        with self.driver.session() as session:

            # 1. Stile del blog
            style_result = session.run("""
                MATCH (s:BlogStyle)
                RETURN s.tono               AS tono,
                       s.registro           AS registro,
                       s.lunghezza_target   AS lunghezza,
                       s.audience           AS audience,
                       s.note_stilistiche   AS note
                LIMIT 1
            """)
            style_record = style_result.single()
            style = dict(style_record) if style_record else {}

            # 2. Claim degli ultimi N post (escluso il topic corrente)
            claim_result = session.run(
                """
                MATCH (p:Post)-[:PARLA_DI]->(r:Ricetta)
                WHERE toLower(r.name) <> toLower($topic)
                WITH p, r
                ORDER BY p.data DESC
                LIMIT $limit
                MATCH (p)-[:HA_CLAIM]->(c:Claim)
                RETURN r.name AS topic_post,
                       p.titolo AS titolo_post,
                       collect(c.testo) AS claims
                """,
                topic=topic_corrente.strip(),
                limit=n_post,
            )

            claim_per_post = []
            for record in claim_result:
                claim_per_post.append(
                    {
                        "topic": record["topic_post"],
                        "titolo": record["titolo_post"],
                        "claims": record["claims"],
                    }
                )

            # 3. Topic correlati per ingredienti condivisi (connessioni tematiche)
            correlati_result = session.run(
                """
                MATCH (r_corrente:Ricetta { name: $topic })-[:CONTIENE]->(i:Ingrediente)
                      <-[:CONTIENE]-(r_altro:Ricetta)<-[:PARLA_DI]-(p:Post)
                WHERE toLower(r_altro.name) <> toLower($topic)
                RETURN DISTINCT r_altro.name AS topic_correlato,
                                collect(DISTINCT i.name)[..3] AS ingredienti_comuni
                LIMIT 3
                """,
                topic=topic_corrente.strip().capitalize(),
            )

            correlati = []
            for record in correlati_result:
                correlati.append(
                    {
                        "topic": record["topic_correlato"],
                        "ingredienti_comuni": record["ingredienti_comuni"],
                    }
                )

        print(
            f"[NEO4J] Contesto editoriale recuperato: "
            f"style={'sì' if style else 'no'}, "
            f"claim_post={len(claim_per_post)}, "
            f"correlati={len(correlati)}"
        )

        return {
            "style": style,
            "claim_correlati": claim_per_post,
            "topic_correlati": correlati,
        }

    def get_ultimi_post_pubblicati(self, limite: int = 5):
        """
        Recupera gli ultimi N post pubblicati dal blog, ordinati dal più recente.
        Restituisce una lista di dizionari con titolo, data e ricetta trattata.
        """
        query_cypher = """
        MATCH (p:Post)-[:PARLA_DI]->(r:Ricetta)
        RETURN p.titolo AS titolo, p.data AS data, r.name AS topic_trattato
        ORDER BY p.data DESC
        LIMIT $limite
        """

        with self.driver.session() as session:
            risultati = session.run(query_cypher, limite=limite)

            ultimi_post = []
            for record in risultati:
                ultimi_post.append(
                    {
                        "titolo": record["titolo"],
                        "data": record["data"],
                        "topic_trattato": record["topic_trattato"],
                    }
                )
            print(f"[NEO4J] Recuperati {ultimi_post} ultimi post pubblicati.")
            return ultimi_post

    def controlla_cronologia_post(self, topic: str):
        """
        Verifica semanticamente se il piatto o una sua variante ha già un Post associato.
        Ritorna un dizionario con i dettagli se trova un duplicato semantico, altrimenti None.
        """
        topic_pulito = topic.strip()

        vettore_topic = self.embeddings.embed_query(topic_pulito)

        # . Vector Search tramite Cypher

        query_cypher = """
        CALL db.index.vector.queryNodes('recipe_embedding_index', 3, $vettore)
        YIELD node AS r, score
        WHERE score > 0.95
        MATCH (p:Post)-[:PARLA_DI]->(r)
        RETURN r.name AS piatto_trattato, p.titolo AS titolo_post, score
        ORDER BY score DESC
        LIMIT 1
        """

        with self.driver.session() as session:
            risultati = session.run(query_cypher, vettore=vettore_topic)
            record = risultati.single()

            if record and record["titolo_post"] is not None:
                #
                print(f"\n [NEO4J K-RAG] Blocco Semantico attivato!")
                print(f"    L'agente ha proposto: '{topic_pulito}'")
                print(
                    f"    Trovato nel DB: '{record['piatto_trattato']}' (Similarità: {record['score']:.2f})"
                )

                return {
                    "piatto_trattato": record["piatto_trattato"],
                    "titolo_post": record["titolo_post"],
                }

        return None

    def espandi_query_per_krag(self, topic: str):
        """
        Estrae gli ingredienti correlati dal Grafo per arricchire la query RAG o generare varianti.
        Utilizza la ricerca semantica K-RAG per trovare la ricetta anche in caso di sinonimi.
        """
        topic_pulito = topic.strip()

        # 1. Calcoliamo il vettore del topic
        vettore_topic = self.embeddings.embed_query(topic_pulito)

        # 2. Cerchiamo la ricetta nello spazio vettoriale, POI attraversiamo il grafo per gli ingredienti
        query_cypher = """
        CALL db.index.vector.queryNodes('recipe_embedding_index', 1, $vettore)
        YIELD node AS r, score
        WHERE score > 0.85
        MATCH (r)-[:CONTIENE]->(i:Ingrediente)
        RETURN i.name AS ingrediente
        """

        termini_espansi = []
        with self.driver.session() as session:
            risultati = session.run(query_cypher, vettore=vettore_topic)
            for record in risultati:
                if record["ingrediente"]:
                    termini_espansi.append(record["ingrediente"])

        return termini_espansi

    def salva_post(
        self,
        topic_originale: str,
        topic_finale: str,
        ingredienti_diretti: list | None = None,
        sotto_ricette: list | None = None,
        fonte: list[str] | None = None,
        testo_post: str = "",
        llm=None,
    ):
        ingredienti_diretti = ingredienti_diretti or []
        sotto_ricette = sotto_ricette or []

        op_madre = topic_originale.strip().capitalize()
        op_finale = topic_finale.strip().capitalize()
        data_oggi = datetime.now().strftime("%Y-%m-%d")
        titolo_post = f"Post su {op_finale} - {data_oggi}"

        is_variante = op_madre.lower() != op_finale.lower()

        print(f"{op_madre.lower()}, {op_finale.lower()}")

        # ==================================================
        # LA MAGIA K-RAG: Calcolo dei vettori prima del salvataggio
        # ==================================================
        vettore_madre = self.embeddings.embed_query(op_madre)
        vettore_finale = self.embeddings.embed_query(op_finale)

        with self.driver.session() as session:

            # BLOG ROOT
            session.run("""
                MERGE (b:Blog { name: "Il mio Blog di Cucina" })
            """)

            # RICETTA + POST
            if is_variante:
                session.run(
                    """
                    MERGE (b:Blog { name: "Il mio Blog di Cucina" })
 
                    MERGE (madre:Ricetta { name: $topic_originale })
                    SET madre.embedding = $vettore_madre
 
                    MERGE (variante:Ricetta { name: $topic_finale })
                    SET variante.embedding = $vettore_finale
 
                    MERGE (variante)-[:IS_VARIANTE_DI]->(madre)
 
                    CREATE (p:Post { titolo: $titolo, data: $data })
                    MERGE (b)-[:HA_PUBBLICATO]->(p)
                    MERGE (p)-[:PARLA_DI]->(variante)
                    """,
                    topic_originale=op_madre,
                    topic_finale=op_finale,
                    titolo=titolo_post,
                    data=data_oggi,
                    vettore_madre=vettore_madre,
                    vettore_finale=vettore_finale,
                )

                session.run(
                    """
                    MATCH (p:Post { titolo: $titolo })
                    MATCH (s:BlogStyle { name: "stile_blog" })
                    MERGE (p)-[:SCRITTO_CON]->(s)
                    """,
                    titolo=titolo_post,
                )
                print(
                    f"[NEO4J] Variante '{op_finale}' -> '{op_madre}' salvata con vettori."
                )

            else:
                session.run(
                    """
                    MERGE (b:Blog { name: "Il mio Blog di Cucina" })
 
                    MERGE (r:Ricetta { name: $topic_originale })
                    SET r.embedding = $vettore_madre
 
                    CREATE (p:Post { titolo: $titolo, data: $data })
                    MERGE (b)-[:HA_PUBBLICATO]->(p)
                    MERGE (p)-[:PARLA_DI]->(r)
                    """,
                    topic_originale=op_madre,
                    titolo=titolo_post,
                    data=data_oggi,
                    vettore_madre=vettore_madre,
                )

                session.run(
                    """
                    MATCH (p:Post { titolo: $titolo })
                    MATCH (s:BlogStyle { name: "stile_blog" })
                    MERGE (p)-[:SCRITTO_CON]->(s)
                    """,
                    titolo=titolo_post,
                )
                print(f"[NEO4J] Ricetta standard '{op_madre}' salvata con vettori.")

            # ==================================================
            # INGREDIENTI DIRETTI
            # ==================================================

            if ingredienti_diretti:

                session.run(
                    """
                    MATCH (r:Ricetta {
                        name: $topic_finale
                    })
 
                    UNWIND $ingredienti AS ing
 
                    MERGE (i:Ingrediente {
                        name: toLower(trim(ing.nome))
                    })
 
                    MERGE (r)-[rel:CONTIENE]->(i)
 
                    SET rel.quantita = ing.quantita
                    """,
                    topic_finale=op_finale,
                    ingredienti=ingredienti_diretti,
                )

                print(
                    f"[NEO4J] "
                    f"{len(ingredienti_diretti)} "
                    f"ingredienti diretti salvati."
                )

            # ==================================================
            # SOTTORICETTE
            # ==================================================

            for sub in sotto_ricette:
                sub_classe = sub["classe_astratta"].strip().lower()
                sub_nome = sub["nome_specifico"].strip().lower()

                # Calcolo embedding
                vettore_specifico = self.embeddings.embed_query(sub_nome)
                vettore_astratto = (
                    self.embeddings.embed_query(sub_classe)
                    if sub_nome != sub_classe
                    else None
                )

                session.run(
                    """
                    MATCH (main:Ricetta { name: $topic_finale })
                    MERGE (madre_sub:Ricetta { name: $classe_astratta })
                    SET madre_sub.embedding = $v_astratto  
                    MERGE (specifica_sub:Ricetta { name: $nome_specifico })
                    SET specifica_sub.embedding = $v_specifico  
                    MERGE (main)-[:USA_PREPARAZIONE]->(specifica_sub)
                    """,
                    topic_finale=op_finale,
                    nome_specifico=sub_nome,
                    classe_astratta=sub_classe,
                    v_specifico=vettore_specifico,
                    v_astratto=vettore_astratto,
                )

                if sub_nome != sub_classe:
                    session.run(
                        """
                        MERGE (specifica_sub:Ricetta { name: $nome_specifico })
                        MERGE (madre_sub:Ricetta { name: $classe_astratta })
                        MERGE (specifica_sub)-[:IS_VARIANTE_DI]->(madre_sub)
                        """,
                        nome_specifico=sub_nome,
                        classe_astratta=sub_classe,
                    )
                    print(
                        f"[NEO4J] Registrata variante interna: '{sub_nome}' -> '{sub_classe}'"
                    )

            if sotto_ricette:

                print(f"[NEO4J] " f"{len(sotto_ricette)} " f"sottoricette salvate.")

            if fonte:
                for url in fonte:
                    if url and url.strip():
                        session.run(
                            """
                    MATCH (p:Post { titolo: $titolo })
                    MERGE (f:Fonte { url: $url })
                    MERGE (p)-[:USA_FONTE]->(f)
                    """,
                            titolo=titolo_post,
                            url=url.strip(),
                        )
                print(f"[NEO4J] {len(fonte)} fonti registrate.")

            # ==================================================
            # CLAIM — estratti dal testo del post tramite LLM
            # ==================================================
            if testo_post and llm:
                claim_estratti = self._estrai_claim(testo_post, topic_finale, llm)

                if claim_estratti:
                    session.run(
                        """
                        MATCH (p:Post { titolo: $titolo })
                        UNWIND $claims AS testo_claim
                        CREATE (c:Claim { testo: testo_claim })
                        MERGE (p)-[:HA_CLAIM]->(c)
                        """,
                        titolo=titolo_post,
                        claims=claim_estratti,
                    )
                    print(
                        f"[NEO4J] {len(claim_estratti)} claim salvati per '{titolo_post}'."
                    )
                else:
                    print(
                        "[NEO4J] Nessun claim estratto (testo vuoto o LLM non disponibile)."
                    )

        print(f"[NEO4J] Post '{titolo_post}' " f"salvato correttamente.")

    def _estrai_claim(self, testo_post: str, topic: str, llm) -> list[str]:
        """
        Usa l'LLM per estrarre 3-5 claim chiave dal testo del post.
        I claim sono affermazioni fattuali brevi che possono essere richiamate
        nei post successivi per creare coerenza editoriale.
        """
        prompt = f"""
        Leggi il seguente post di un blog di cucina sul tema '{topic}'.
        Estrai esattamente 3-5 CLAIM CHIAVE: affermazioni fattuali, tecniche o culturali
        brevi (max 25 parole ciascuna) che potrebbero essere richiamate o referenziate
        in futuri post per creare coerenza editoriale.

        Esempi di claim ben formati:
        - "La besciamella richiede un roux di burro e farina in parti uguali."
        - "La pasta alla norma è un primo piatto tradizionale catanese con melanzane fritte."
        - "Il ragù siciliano si distingue da quello bolognese per l'uso del vino bianco."

        Rispondi SOLO con una lista Python valida di stringhe, senza altro testo.
        Esempio: ["claim 1", "claim 2", "claim 3"]

        POST:
        {testo_post[:3000]}
        """
        try:
            risposta = llm.invoke([HumanMessage(content=prompt)])
            testo = risposta.content.strip()

            # Pulizia robusta: togliamo eventuali backtick o prefissi
            testo = testo.replace("```python", "").replace("```", "").strip()

            # Parsing della lista
            import ast

            claim_list = ast.literal_eval(testo)

            if isinstance(claim_list, list):
                # Filtriamo claim troppo lunghi o vuoti
                return [
                    c.strip()
                    for c in claim_list
                    if isinstance(c, str) and 5 < len(c) < 200
                ]

        except Exception as e:
            print(f"[NEO4J] Errore estrazione claim: {e}")

        return []

    def __del__(self):
        """Si attiva automaticamente quando l'oggetto viene rimosso dalla memoria."""
        if hasattr(self, "driver") and self.driver:
            print(" [NEO4J] Chiusura sicura del driver di connessione.")
            self.driver.close()


# Esportiamo l'istanza pronta all'uso
kg_client = CucinaKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
