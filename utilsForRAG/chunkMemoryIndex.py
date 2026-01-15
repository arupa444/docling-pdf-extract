import faiss
import numpy as np

class ChunkMemoryIndex:
    def __init__(self, dim=768):  # Gemini embeddings are usually 768 dimensions
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids = []

    def add(self, chunk_id, embedding):
        # Normalize for Cosine Similarity if using IndexFlatIP (Inner Product)
        # Gemini embeddings are usually normalized, but good practice to ensure.
        vec = np.array([embedding]).astype("float32")
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.chunk_ids.append(chunk_id)

    def search(self, query_embedding, k=3):
        vec = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(vec)

        scores, ids = self.index.search(vec, k)

        results = []
        for idx, i in enumerate(ids[0]):
            if i != -1:  # FAISS returns -1 if not enough neighbors found
                results.append((self.chunk_ids[i], float(scores[0][idx])))
        return results
