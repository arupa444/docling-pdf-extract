from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Answer:
    def answer(query, retrieved_chunks, llm):
        evidence_text = "\n\n".join(
            f"SOURCE ID: {c['chunk_id']}\nEVIDENCE:\n" + "\n".join(f"- {p}" for p in c["evidence"])
            for c in retrieved_chunks
        )

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            Answer the user's question using ONLY the provided evidence. 
            Cite the SOURCE ID for every fact you use.
            If you cannot answer based on the evidence, say so.
            """),
            ("user", "Question: {query}\n\nEvidence:\n{evidence}")
        ])

        runnable = PROMPT | llm | StrOutputParser()
        return runnable.invoke({"query": query, "evidence": evidence_text})
