import os
from neo4j import GraphDatabase
from datetime import datetime


class CucinaKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

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
        Verifica se il piatto o una sua variante ha già un Post associato.
        Ritorna un dizionario con i dettagli se trova un duplicato, altrimenti None.
        """

        topic_pulito = topic.strip().lower()
        query_cypher = """
    MATCH (r:Ricetta)
    WHERE toLower(r.name) = toLower($topic)
    MATCH (p:Post)-[:PARLA_DI]->(r)
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
        ingredienti_diretti: list | None = None,
        sotto_ricette: list | None = None,
        fonte: str = "",
    ):

        ingredienti_diretti = ingredienti_diretti or []
        sotto_ricette = sotto_ricette or []

        op_madre = topic_originale.strip().capitalize()
        op_finale = topic_finale.strip().capitalize()

        data_oggi = datetime.now().strftime("%Y-%m-%d")

        titolo_post = f"Post su {op_finale} - {data_oggi}"

        is_variante = op_madre.lower() != op_finale.lower()

        with self.driver.session() as session:

            # ==================================================
            # BLOG ROOT
            # ==================================================

            session.run("""
                MERGE (b:Blog {
                    name: "Il mio Blog di Cucina Siciliana"
                })
                """)

            # ==================================================
            # RICETTA + POST
            # ==================================================

            if is_variante:

                session.run(
                    """
                    MERGE (b:Blog {
                        name: "Il mio Blog di Cucina Siciliana"
                    })

                    MERGE (madre:Ricetta {
                        name: $topic_originale
                    })

                    MERGE (variante:Ricetta {
                        name: $topic_finale
                    })

                    MERGE (variante)-[:IS_VARIANTE_DI]->(madre)

                    CREATE (p:Post {
                        titolo: $titolo,
                        data: $data
                    })

                    MERGE (b)-[:HA_PUBBLICATO]->(p)

                    MERGE (p)-[:PARLA_DI]->(variante)
                    """,
                    topic_originale=op_madre,
                    topic_finale=op_finale,
                    titolo=titolo_post,
                    data=data_oggi,
                )

                print(f"[NEO4J] Variante " f"'{op_finale}' -> '{op_madre}'")

            else:

                session.run(
                    """
                    MERGE (b:Blog {
                        name: "Il mio Blog di Cucina Siciliana"
                    })

                    MERGE (r:Ricetta {
                        name: $topic_originale
                    })

                    CREATE (p:Post {
                        titolo: $titolo,
                        data: $data
                    })

                    MERGE (b)-[:HA_PUBBLICATO]->(p)

                    MERGE (p)-[:PARLA_DI]->(r)
                    """,
                    topic_originale=op_madre,
                    titolo=titolo_post,
                    data=data_oggi,
                )

                print(f"[NEO4J] Ricetta standard " f"'{op_madre}'")

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

                session.run(
                    """
                    
                        MATCH (main:Ricetta {
                            name: $topic_finale
                        })
                        MERGE (madre_sub:Ricetta {
                            name: toLower($classe_astratta)
                        })
                        MERGE (specifica_sub:Ricetta {
                            name: toLower($nome_specifico)
                        })
                        MERGE (specifica_sub)-[:IS_VARIANTE_DI]->(madre_sub)

                        MERGE (main)-[:USA_PREPARAZIONE]->(specifica_sub)
                        """,
                    topic_finale=op_finale,
                    nome_specifico=sub["nome_specifico"],
                    classe_astratta=sub["classe_astratta"],
                )

                ingredienti_sub = sub.get("ingredienti", [])

                if ingredienti_sub:

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
                        nome_specifico=sub["nome_specifico"],
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
