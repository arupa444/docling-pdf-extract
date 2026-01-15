class Retrieve:
    def retrieve(query, chunker, memory_index):
        print(f"\n[bold magenta]Searching for:[/bold magenta] '{query}'")

        # Embed the query
        query_embedding = chunker.embedder.embed_query(query)

        # Search Index
        results = memory_index.search(query_embedding)

        # Hydrate results from Chunker memory
        return [{
            "chunk_id": cid,
            "score": score,
            "title": chunker.chunks[cid]["title"],
            "summary": chunker.chunks[cid]["summary"],
            "evidence": chunker.chunks[cid]["propositions"],
            "full_text": chunker.chunks[cid]["canonical_text"]
        } for cid, score in results]
