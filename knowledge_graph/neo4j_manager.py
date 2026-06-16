import os
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from datetime import datetime


class CucinaKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        print("[NEO4J] Inizializzazione embeddings locali per il Grafo...")
        # Usiamo lo STESSO modello di ChromaDB.
        # Questo modello produce vettori a 384 dimensioni.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Creiamo l'indice vettoriale
        self._crea_indice_vettoriale()

    def _crea_indice_vettoriale(self):
        """Prepara il database a grafo per la ricerca semantica K-RAG."""
        query = """
        CREATE VECTOR INDEX recipe_embedding_index IF NOT EXISTS
        FOR (r:Ricetta) ON (r.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 384,
          `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)
            print(
                " [NEO4J] Indice vettoriale 'recipe_embedding_index' verificato/creato."
            )

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
        WHERE score > 0.88
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
        fonte: str = "",
    ):
        ingredienti_diretti = ingredienti_diretti or []
        sotto_ricette = sotto_ricette or []

        op_madre = topic_originale.strip().capitalize()
        op_finale = topic_finale.strip().capitalize()
        data_oggi = datetime.datetime.now().strftime("%Y-%m-%d")
        titolo_post = f"Post su {op_finale} - {data_oggi}"

        is_variante = op_madre.lower() != op_finale.lower()

        # ==================================================
        # LA MAGIA K-RAG: Calcolo dei vettori prima del salvataggio
        # ==================================================
        vettore_madre = self.embeddings.embed_query(op_madre)
        vettore_finale = self.embeddings.embed_query(op_finale)

        with self.driver.session() as session:

            # BLOG ROOT
            session.run("""
                MERGE (b:Blog { name: "Il mio Blog di Cucina Siciliana" })
            """)

            # RICETTA + POST
            if is_variante:
                session.run(
                    """
                    MERGE (b:Blog { name: "Il mio Blog di Cucina Siciliana" })

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
                print(
                    f"[NEO4J] Variante '{op_finale}' -> '{op_madre}' salvata con vettori."
                )

            else:
                session.run(
                    """
                    MERGE (b:Blog { name: "Il mio Blog di Cucina Siciliana" })

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
                nome_specifico = sub["nome_specifico"]
                classe_astratta = sub["classe_astratta"]

                # 1. Calcoliamo i vettori per la sottoricetta
                vettore_specifico = self.embeddings.embed_query(nome_specifico)
                vettore_astratto = self.embeddings.embed_query(classe_astratta)

                session.run(
                    """
                    MATCH (main:Ricetta {
                        name: $topic_finale
                    })
                    
                    // Salviamo la classe astratta con il suo vettore
                    MERGE (madre_sub:Ricetta {
                        name: toLower($classe_astratta)
                    })
                    SET madre_sub.embedding = $vettore_astratto
                    
                    // Salviamo la variante specifica con il suo vettore
                    MERGE (specifica_sub:Ricetta {
                        name: toLower($nome_specifico)
                    })
                    SET specifica_sub.embedding = $vettore_specifico
                    
                    MERGE (specifica_sub)-[:IS_VARIANTE_DI]->(madre_sub)
                    MERGE (main)-[:USA_PREPARAZIONE]->(specifica_sub)
                    """,
                    topic_finale=op_finale,
                    nome_specifico=nome_specifico,
                    classe_astratta=classe_astratta,
                    vettore_specifico=vettore_specifico,  # Passiamo i vettori
                    vettore_astratto=vettore_astratto,  # Passiamo i vettori
                )

                ingredienti_sub = sub.get("ingredienti", [])

                if ingredienti_sub:
                    # Il salvataggio degli ingredienti della sottoricetta rimane testuale e strutturale
                    session.run(
                        """
                        MATCH (specifica_sub:Ricetta {
                            name: toLower($nome_specifico)
                        })
                        UNWIND $ingredienti AS ing
                        MERGE (i:Ingrediente {
                            name: toLower(trim(ing.nome))
                        })
                        MERGE (specifica_sub)-[rel:CONTIENE]->(i)
                        SET rel.quantita = ing.quantita
                        SET rel.fase = ing.fase_utilizzo
                        """,
                        nome_specifico=nome_specifico,
                        ingredienti=ingredienti_sub,
                    )

            if sotto_ricette:

                print(f"[NEO4J] " f"{len(sotto_ricette)} " f"sottoricette salvate.")

            if fonte:

                session.run(
                    """
                    MATCH (p:Post {
                        titolo: $titolo
                    })

                    MERGE (f:Fonte {
                        url: $fonte
                    })

                    MERGE (p)-[:USA_FONTE]->(f)
                    """,
                    titolo=titolo_post,
                    fonte=fonte,
                )

                print(f"[NEO4J] Fonte registrata.")

        print(f"[NEO4J] Post '{titolo_post}' " f"salvato correttamente.")

    def __del__(self):
        """Si attiva automaticamente quando l'oggetto viene rimosso dalla memoria."""
        if hasattr(self, "driver") and self.driver:
            print(" [NEO4J] Chiusura sicura del driver di connessione.")
            self.driver.close()


# Esportiamo l'istanza pronta all'uso
kg_client = CucinaKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
