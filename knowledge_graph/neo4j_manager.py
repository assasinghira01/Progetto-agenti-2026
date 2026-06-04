import os
from neo4j import GraphDatabase
from datetime import datetime


class CucinaKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def controlla_cronologia_post(self, topic: str):
        """
        Verifica se il piatto o una sua variante ha già un Post associato.
        Ritorna un dizionario con i dettagli se trova un duplicato, altrimenti None.
        """

        topic_pulito = topic.strip().lower()
        query_cypher = """
    // 1. Cerchiamo la ricetta che corrisponde al testo inserito
    MATCH (r:Ricetta)
    WHERE toLower(r.name) = toLower($topic)
    
    // 2. Cerchiamo SOLO il post direttamente collegato a questo specifico nodo
    MATCH (p:Post)-[:PARLA_DI]->(r)
    
    // 3. Restituiamo i dati del post diretto
    RETURN r.name AS piatto_trattato, p.titolo AS titolo_post
    LIMIT 1
    """
        with self.driver.session() as session:
            risultati = session.run(query_cypher, topic=topic_pulito)
            record = risultati.single()
            if record and record["titolo_post"] is not None:
                return {
                    "piatto_trattato": record["piatto_trattato"],
                    "titolo_post": record["titolo_post"],
                }
        return None

    def espandi_query_per_krag(self, topic: str):
        """
        Estrae gli ingredienti correlati dal Grafo per arricchire la query RAG.
        Es: Se cerchi 'Caponata', estrae ['Melanzane', 'Sedano', 'Agrodolce'].
        """
        query_cypher = """
        MATCH (r:Ricetta)-[:CONTIENE]->(i:Ingrediente)
        WHERE r.name =~ ('(?i)' + $topic)
        RETURN i.name AS ingrediente
        """
        termini_espansi = []
        with self.driver.session() as session:
            risultati = session.run(query_cypher, topic=topic)
            for record in risultati:
                if record["ingrediente"]:
                    termini_espansi.append(record["ingrediente"])
        return termini_espansi

    def salva_post(
        self,
        topic_originale: str,
        topic_finale: str,
        ingredienti: list = None,
        fonti: list = None,
        claims: list = None,
    ):

        ingredienti = ingredienti or []
        fonti = fonti or []
        claims = claims or []

        op_madre = topic_originale.strip().capitalize()
        op_finale = topic_finale.strip().capitalize()
        data_oggi = datetime.now().strftime("%Y-%m-%d")
        titolo_post = f"Post su {op_finale} - {data_oggi}"
        is_variante = op_madre.lower() != op_finale.lower()

        with self.driver.session() as session:

            # ── FASE 0: Nodo radice Blog (hub comune a tutto il grafo) ──────────
            session.run("""
            MERGE (b:Blog {name: "Il mio Blog di Cucina Siciliana"})
        """)

            # ── FASE 1: Struttura concettuale ───────────────────────────────────
            if is_variante:
                session.run(
                    """
                MERGE (b:Blog {name: "Il mio Blog di Cucina Siciliana"})
                MERGE (madre:Ricetta {name: $topic_originale})
                MERGE (variante:Ricetta {name: $topic_finale})
                MERGE (variante)-[:IS_VARIANTE_DI]->(madre)
                CREATE (p:Post {name: $titolo, titolo: $titolo, data: $data})
                CREATE (b)-[:HA_PUBBLICATO]->(p)  
                CREATE (p)-[:PARLA_DI]->(variante)
                    """,
                    topic_originale=op_madre,
                    topic_finale=op_finale,
                    titolo=titolo_post,
                    data=data_oggi,
                )
                print(f"[NEO4J] Variante '{op_finale}' → '{op_madre}'")
            else:
                session.run(
                    """
                MERGE (b:Blog {name: "Il mio Blog di Cucina Siciliana"})
                MERGE (r:Ricetta {name: $topic_originale})
                CREATE (p:Post {name: $titolo, titolo: $titolo, data: $data})
                CREATE (b)-[:HA_PUBBLICATO]->(p)  
                CREATE (p)-[:PARLA_DI]->(r)
                    """,
                    topic_originale=op_madre,
                    titolo=titolo_post,
                    data=data_oggi,
                )
            print(f"[NEO4J] Ricetta standard '{op_madre}'")

            # ── FASE 2: Ingredienti (collegati al topic_finale, non solo madre) ─
            # FIX: usiamo topic_finale così funziona anche per le varianti
            if ingredienti:
                session.run(
                    """
                MATCH (r:Ricetta {name: $topic_finale})
                UNWIND $ingredienti AS ing_name
                MERGE (i:Ingrediente {name: ing_name})
                MERGE (r)-[:CONTIENE]->(i)
            """,
                    topic_finale=op_finale,
                    ingredienti=ingredienti,
                )
            print(f"[NEO4J] {len(ingredienti)} ingredienti → '{op_finale}'")

        print(f"[NEO4J] Post '{titolo_post}' salvato correttamente.")

    def __del__(self):
        """Si attiva automaticamente quando l'oggetto viene rimosso dalla memoria."""
        if hasattr(self, "driver") and self.driver:
            print(" [NEO4J] Chiusura sicura del driver di connessione.")
            self.driver.close()


# Esportiamo l'istanza pronta all'uso
kg_client = CucinaKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
