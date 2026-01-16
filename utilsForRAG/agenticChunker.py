import os
import uuid
import json
import math
from typing import List
from rich import print
from rich.console import Console
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings




console = Console()



class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=api_key,
            temperature=0
        )

        # STEP 2: Embeddings initialized here
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

    def cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

    # --- PART 1: PROPOSITION GENERATION ---
    def generate_propositions(self, text: str | dict) -> List[str]:
        if self.print_logging:
            console.print("[bold blue]Generating propositions...[/bold blue]")

        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
    You are a high-precision information decomposition engine.

    OBJECTIVE:
    Decompose the input into COMPLETE, ATOMIC propositions with ZERO information loss.

    INPUT MAY BE:
    - Plain text
    - Markdown
    - Python dictionary / JSON
    - Dictionary containing multiple sources (files, websites, PDFs)
    - Mixed content including tables, code blocks, and scraped text

    STRICT RULES (NON-NEGOTIABLE):
    - DO NOT omit, merge, summarize, generalize, or infer information
    - EVERY fact, table row, code behavior, config value, and nested key MUST appear
    - Treat EACH dictionary key-path as a separate logical source
    - Tables: extract EACH ROW as one or more propositions
    - Code blocks: extract functional and semantic behavior (not syntax-only summaries)
    - Lists: each bullet = at least one proposition
    - Redundant information MUST be kept
    - If something is unclear, corrupted, or truncated, still create a proposition and FLAG it

    PROPOSITION RULES:
    - Each proposition must express ONE atomic fact
    - Propositions must be self-contained and unambiguous
    - Preserve original meaning exactly
    - Do NOT add new information

    OUTPUT FORMAT (MANDATORY):
    Return STRICTLY a valid JSON array of strings.
    No markdown. No explanations. No surrounding text.

    FINAL CHECK:
    Before responding, verify that EVERY part of the input is represented by at least one proposition.
    If anything is missing, STOP and re-process.
                """
            ),
            (
                "user",
                """
    <INPUT_PAYLOAD>
    {text}
    </INPUT_PAYLOAD>
                """
            )
        ])

        runnable = PROMPT | self.llm | StrOutputParser()
        raw_response = runnable.invoke({"text": text})
        cleaned_response = (
            raw_response
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        try:
            parsed = json.loads(cleaned_response)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                print(parsed)
                return parsed
            else:
                raise ValueError("Parsed JSON is not a list of strings")

        except Exception:
            return [
                line.strip("-• ").strip()
                for line in cleaned_response.splitlines()
                if line.strip()
            ]

    # --- PART 2: THE CHUNKING LOGIC ---

    # ✅ STEP 1: Canonicalization Method
    def _canonicalize_chunk(self, chunk_data):
        facts = "\n".join(f"- {p}" for p in chunk_data["propositions"])
        return f"""
        TITLE: {chunk_data['title']}
        SUMMARY: {chunk_data['summary']}
        FACTS:
        {facts}
        """.strip()

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition):
        if self.print_logging:
            print(f"\nProcessing: '[italic]{proposition}[/italic]'")

        if len(self.chunks) == 0:
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        if chunk_id:
            if self.print_logging:
                print(f"[bold green]Chunk Found[/bold green] ({chunk_id}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
        else:
            if self.print_logging:
                print("[bold yellow]No relevant chunk found. Creating a new one.[/bold yellow]")
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)

        # Update summary/title
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

        # ✅ STEP 2 Update: Canonicalize & Embed on update
        self.chunks[chunk_id]["canonical_text"] = self._canonicalize_chunk(self.chunks[chunk_id])
        self.chunks[chunk_id]["embedding"] = self.embedder.embed_query(self.chunks[chunk_id]["canonical_text"])

    def _create_new_chunk(self, proposition):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        # Create temporary dict to generate canonical text
        chunk_data = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }

        # ✅ STEP 2 Update: Canonicalize & Embed on creation
        canonical_text = self._canonicalize_chunk(chunk_data)
        embedding = self.embedder.embed_query(canonical_text)

        # Finalize chunk storage
        chunk_data['canonical_text'] = canonical_text
        chunk_data['embedding'] = embedding

        self.chunks[new_chunk_id] = chunk_data

        if self.print_logging:
            print(f"Created chunk ({new_chunk_id}): [bold]{new_chunk_title}[/bold]")

    # --- PART 3: DECISION MAKING AGENTS ---
    def _llm_judge_chunk(self, proposition, candidate_chunk_ids):
        outline = ""
        for cid in candidate_chunk_ids:
            c = self.chunks[cid]
            outline += f"Chunk ID: {cid}\nSummary: {c['summary']}\n\n"

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            Decide if the proposition belongs to any chunk below.
            Return ONLY the chunk_id or "No chunks".
            """),
            ("user", "Chunks:\n{outline}\nProposition:\n{proposition}")
        ])

        response = (PROMPT | self.llm | StrOutputParser()).invoke({
            "outline": outline,
            "proposition": proposition
        }).strip()

        return response if response in candidate_chunk_ids else None

    def _find_relevant_chunk(self, proposition):
        prop_embedding = self.embedder.embed_query(proposition)
        scored_chunks = []
        for cid, chunk in self.chunks.items():
            # Uses the embedding of the Canonical Text now (much more accurate)
            score = self.cosine_similarity(prop_embedding, chunk['embedding'])
            scored_chunks.append((cid, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [cid for cid, score in scored_chunks[:3] if score > 0.75]

        if not top_candidates:
            return None

        return self._llm_judge_chunk(proposition, top_candidates)

    # --- PART 4: SUMMARIZATION AGENTS ---
    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Create a short 1-sentence summary for a new content group containing this proposition."),
            ("user", "{proposition}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({"proposition": proposition})

    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Create a very brief (2-4 words) title for a content group with this summary."),
            ("user", "{summary}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({"summary": summary})

    def _update_chunk_summary(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system",
             "Update the summary for this chunk based on the existing summary and the propositions. Keep it 1 sentence."),
            ("user", "Propositions:\n{propositions}\n\nCurrent Summary:\n{current_summary}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({
            "propositions": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        })

    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Update the brief title based on the summary. Return ONLY the title."),
            ("user", "Summary:\n{current_summary}\n\nCurrent Title:\n{current_title}")
        ])
        return (PROMPT | self.llm | StrOutputParser()).invoke({
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        })

    def load_chunks(self, path: str):
        """
        Loads previously saved chunks from the JSON file.
        """
        if not os.path.exists(path):
            print(f"[bold red]File not found:[/bold red] {path}")
            return False

        with open(path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"[bold green]Successfully loaded {len(self.chunks)} chunks from disk.[/bold green]")
        return True

    def pretty_print_chunks(self):
        print(f"\n[bold magenta]Final Results: {len(self.chunks)} Chunks Created[/bold magenta]\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"[bold]Chunk ({chunk_id})[/bold]: {chunk['title']}")
