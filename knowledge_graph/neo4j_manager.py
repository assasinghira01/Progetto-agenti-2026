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
        query_cypher = """
        MATCH (r:Ricetta) WHERE toLower(r.name) CONTAINS toLower($topic)
        OPTIONAL MATCH (r)-[:IS_VARIANTE_DI]-(padre:Concetto)-[:IS_VARIANTE_DI]-(variante:Ricetta)
        MATCH (p:Post)-[:PARLA_DI]->(collegato)
        WHERE collegato = r OR collegato = variante
        RETURN collegato.name AS piatto_trattato, p.titolo AS titolo_post
        LIMIT 1
        """
        with self.driver.session() as session:
            risultati = session.run(query_cypher, topic=topic)
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

    def salva_post(self, topic_originale: str, topic_finale: str):
        """
        Salva il post nel Knowledge Graph. Se il topic_finale è una variante di quello originale,
        costruisce una struttura gerarchica collegando la variante alla ricetta madre.
        """
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(uri, auth=(user, pwd))

        # Controllo se l'output finale differisce dal concetto originale (è una variante)
        if topic_originale.lower() != topic_finale.lower():
            query = """
        // 1. Assicuriamo l'esistenza della Ricetta madre
        MERGE (madre:Ricetta {name: $topic_originale})
        // 2. Creiamo il nodo concettuale della variante
        MERGE (variante:Concetto {name: $topic_finale})
        // 3. Colleghiamo la variante gerarchicamente alla madre
        MERGE (variante)-[:IS_VARIANTE_DI]->(madre)
        // 4. Generiamo il nuovo Post pubblicato
        CREATE (p:Post {data: $data, titolo: $titolo})
        // 5. Colleghiamo il Post all'entità specifica trattata
        CREATE (p)-[:PARLA_DI]->(variante)
        """
            print(
                f"⚙️ [NEO4J] Registrazione variante '{topic_finale}' associata alla radice '{topic_originale}'."
            )
        else:
            # Struttura di pubblicazione standard per ricetta base
            query = """
        MERGE (r:Ricetta {name: $topic_originale})
        CREATE (p:Post {data: $data, titolo: $titolo})
        CREATE (p)-[:PARLA_DI]->(r)
        """
            print(f"⚙️ [NEO4J] Registrazione ricetta standard '{topic_originale}'.")

        with driver.session() as session:
            session.run(
                query,
                topic_originale=topic_originale.capitalize(),
                topic_finale=topic_finale.capitalize(),
                data=datetime.now().strftime("%Y-%m-%d"),
                titolo=f"Post su {topic_finale.capitalize()}",
            )
        driver.close()


# Esportiamo l'istanza pronta all'uso
kg_client = CucinaKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
