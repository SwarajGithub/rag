import os
import psycopg


class DB_Operations:

    def __init__(self, is_age_enabled: bool = False):
        self.dbname=os.environ.get("DB_NAME")
        self.user=os.environ.get("DB_USER")
        self.password=os.environ.get("DB_PASSWORD") or os.environ.get("DB_PASS")
        self.port=os.environ.get("DB_PORT")
        self.host=os.environ.get("DB_HOST")
        self.is_age_enabled = is_age_enabled

        db_config = {
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
        }

        self.conn = psycopg.connect(**db_config)
        if is_age_enabled:
            self._enable_age()

    def _enable_age(self):
        # LOAD and search_path are session settings. Keep this connection open
        # so later inserts still see Apache AGE. create_graph cannot run inside
        # a transaction block, so the setup itself is autocommit.
        self.conn.autocommit = True
        try:
            self.conn.execute("LOAD 'age';")
            # public comes first so application tables are not shadowed by
            # the copy that was created in ag_catalog. AGE functions stay visible.
            self.conn.execute("SET search_path = public, ag_catalog;")
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
                    ("rag_graph",),
                )
                if cursor.fetchone() is None:
                    cursor.execute("SELECT create_graph('rag_graph');")
                    cursor.fetchone()
        finally:
            self.conn.autocommit = False

    def delete_from_db(self, doc_name: str):
        if not doc_name:
            return
        with self.conn.cursor() as cursor:
            cursor.execute('DELETE FROM public.kb_advanced_chunks WHERE document_id = %s;', (doc_name,))
        self.conn.commit()



    def insert_in_db(self, doc_name: str, chunk, embeddings):
        try:
            with self.conn.cursor() as cursor:
                emb_list = embeddings.tolist()
                cursor.execute(
                    """
                    INSERT INTO public.kb_advanced_chunks (document_id, page_num, content_type, content, image_path, embedding, tsv_content) 
                    VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('english', %s))
                    RETURNING id;
                    """,
                    (doc_name, chunk["page"], chunk["type"], chunk["content"], chunk["image_path"], emb_list, chunk["content"])
                )
                return cursor.fetchone()[0]
        finally:
            self.conn.commit()

    def update_in_db(self, ):
        pass

    def execute_graph_insert(self, graph_payload, chunk_id):
        if not self.is_age_enabled:
            return

        try:
            for entity in graph_payload.get("entities", []):
                name = entity["name"].replace("'", "")
                etype = entity["type"].replace("'", "")

                with self.conn.cursor() as cursor:
                    cursor.execute(f"""
                            SELECT * FROM cypher('rag_graph', $$
                                MERGE (e:Entity {{name: '{name}', type: '{etype}'}})
                            $$) as (v agtype);
                        """)
                    cursor.fetchall()

                # 5. Load structural Relationships (Edges) referencing chunk tracking ID
            for rel in graph_payload.get("relationships", []):
                src = rel["source"].replace("'", "")
                tgt = rel["target"].replace("'", "")
                rtype = rel["type"].replace("'", "").upper().replace(" ", "_")

                with self.conn.cursor() as cursor:
                    cursor.execute(f"""
                            SELECT * FROM cypher('rag_graph', $$
                                MATCH (a:Entity {{name: '{src}'}}), (b:Entity {{name: '{tgt}'}})
                                MERGE (a)-[r:{rtype} {{source_chunk_id: {chunk_id}}}]->(b)
                            $$) as (e agtype);
                        """)
                    cursor.fetchall()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            self.conn.close()

    def get_from_db(self, query_vector, doc_name, query, candidate_pool_limit: int = 3):
        try:
            with self.conn.cursor() as cursor:
                # We query the Top 50 vectors and Top 50 keywords, fusing their ranks seamlessly
                rrf_query = """
                WITH vector_matches AS (
                    SELECT id, page_num, content_type, content, image_path,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) as rank
                    FROM public.kb_advanced_chunks
                    WHERE document_id = %s
                    LIMIT 50
                ),
                keyword_matches AS (
                    SELECT id, page_num, content_type, content, image_path,
                           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv_content, plainto_tsquery('english', %s)) DESC) as rank
                    FROM public.kb_advanced_chunks
                    WHERE document_id = %s AND tsv_content @@ plainto_tsquery('english', %s)
                    LIMIT 50
                )
                SELECT 
                    COALESCE(v.page_num, k.page_num) as page,
                    COALESCE(v.content_type, k.content_type) as chunk_type,
                    COALESCE(v.content, k.content) as content,
                    COALESCE(v.image_path, k.image_path) as image_path,
                    (COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0)) as rrf_score
                FROM vector_matches v
                FULL OUTER JOIN keyword_matches k ON v.id = k.id
                ORDER BY rrf_score DESC
                LIMIT %s;
                """
                cursor.execute(rrf_query, (str(query_vector), doc_name, query, doc_name, query, candidate_pool_limit))
                rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "page": row[0],
                    "type": row[1],
                    "content": row[2],
                    "image_path": row[3],
                    "rrf_score": float(row[4])
                })
        except Exception as e:
            print(e)

        return results

    def query_graph(self, chunk_ids_str):
        cypher_query = f"""
                        SELECT * FROM cypher('rag_graph', $$
                            MATCH (a:Entity)-[r]->(b:Entity)
                            WHERE r.source_chunk_id IN [{chunk_ids_str}]
                            RETURN a.name, type(r), b.name
                        $$) as (source_node agtype, rel_type agtype, target_node agtype);
                    """
        graph_contexts = []
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(cypher_query)
                graph_edges = cursor.fetchall()
                for edge in graph_edges:
                    graph_contexts.append(f"Fact: ({edge[0]}) --[{edge[1]}]--> ({edge[2]})")
            return graph_contexts
        except Exception as e:
            print(f"Graph traversal skipped or met schema error: {e}")
        finally:
            self.conn.close()


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1024):
    from openai import OpenAI
    print("LLM call...............")
    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"),
                    api_key=os.environ.get("API_KEY"),
                    )

    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        extra_body={"think": False}
    )
    res = response.choices[0].message.content.strip()
    trace_llm_call = os.environ.get("TRACE_LLM_CALL", False)
    if trace_llm_call:
        print(f"System prompt {system_prompt}")
        print(f"User prompt {user_prompt}")
        print(f"Response {res}")
    return res