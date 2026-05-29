from neo4j import GraphDatabase

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
                    "titolo_post": record["titolo_post"]
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

def salva_post_approvato(self, topic: str):
        # 2. LA FUNZIONE MANCANTE: Crea il nodo Post e lo collega alla Ricetta
        query = """
        MERGE (r:Ricetta {name: $topic})
        CREATE (p:Post {data: $data, titolo: $titolo})
        CREATE (p)-[:PARLA_DI]->(r)
        """
        with self.driver.session() as session:
            session.run(query, 
                        topic=topic, 
                        data=datetime.now().strftime("%Y-%m-%d"), 
                        titolo=f"Post su {topic}")

                        
# Esportiamo l'istanza pronta all'uso
kg_client = CucinaKnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
