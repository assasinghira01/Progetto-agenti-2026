from neo4j import GraphDatabase

class CucinaKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def espandi_query_per_topic(self, topic: str):
        query_cypher = """
        MATCH (r:Ricetta {name: $topic})-[rel]-(entita_collegata)
        RETURN labels(entita_collegata)[0] AS tipo, entita_collegata.name AS nome
        """
        espansioni = []
        with self.driver.session() as session:
            risultati = session.run(query_cypher, topic=topic)
            for record in risultati:
                espansioni.append(record["nome"])
        return espansioni

# Esportiamo l'istanza pronta all'uso
kg_client = CucinaKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")