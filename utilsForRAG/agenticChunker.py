import os
import uuid
import json
import math
from typing import List, Any
from rich import print
from rich.console import Console
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from itertools import batched  # Requires Python 3.12+



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
            temperature=0.1
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
    def generate_propositions(self, text: str | dict | List[Any]) -> List[str]:
        if self.print_logging:
            console.print("[bold blue]Generating propositions...[/bold blue]")

        print("Started generating propositions...")

        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
            You are an information extraction engine.

            Your task is to extract ALL explicit factual information
            from the input and convert it into atomic propositions.

            IMPORTANT:
            - The input format (markdown, dict, list, scraped text, PDF text)
              is NOT information and must NEVER appear in the output.
            - Do NOT describe formatting, structure, headings, or data types.

            RULES:
            - Extract ONLY meaningful content facts
            - Do NOT mention words like "markdown", "json", "dict", "list", "heading"
            - Do NOT describe structure or formatting
            - Do NOT summarize or merge facts
            - Do NOT infer missing information

            EXTRACTION RULES:
            - Each bullet or sentence → one or more propositions
            - Each table row → propositions based on cell meaning
            - Each dictionary key–value pair → propositions about the VALUE, not the key name
            - Code → extract factual behavior, parameters, values, or purpose

            If text is unclear or noisy, still extract the fact and prefix with:
            "UNCLEAR:"

            PROPOSITION RULES:
            - One fact per proposition
            - Self-contained and precise
            - Preserve original meaning

            OUTPUT:
            Return ONLY a valid JSON array of strings.
            No explanations. No metadata. No labels.

            FINAL CHECK:
            Every meaningful statement in the input must appear
            as at least one proposition.
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

        print("Clean response: ",cleaned_response)

        try:
            parsed = json.loads(cleaned_response)
            if isinstance(parsed, list):
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

    def process_accumulated_data(self, data: str | List[Any] | dict )-> List[str]:
        """
        Dispatches data to generate_propositions based on type and size.
        """
        all_propositions = []

        # CASE 1: It is a single String or Dictionary (No iteration needed)
        if isinstance(data, (str, dict)):
            print(f"Processing single {type(data).__name__}...")
            props = self.generate_propositions(data)
            all_propositions.extend(props)

        # CASE 2: It is a List
        elif isinstance(data, list):
            # Subcase A: List is small (<= 3), send it all at once
            if len(data) <= 3:
                print(f"Processing small list (size {len(data)})...")
                props = self.generate_propositions(data)
                all_propositions.extend(props)

            # Subcase B: List is large (> 3), iterate in batches of 3
            else:
                print(f"Batching large list (size {len(data)}) into sets of 3...")

                # MODERN APPROACH: itertools.batched
                for batch_tuple in batched(data, 3):
                    # batched returns a tuple, convert to list for your function
                    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",batch_tuple,"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
                    batch_list = list(batch_tuple)

                    # Pass this chunk of 3 to the LLM
                    props = self.generate_propositions(batch_list)

                    print("Here are the propositions inside the batch:")

                    print(props)

                    all_propositions.extend(props)
        print(f"Alll Propositions size: {len(all_propositions)}")
        print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n{all_propositions}\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        return all_propositions

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

        print("Received chunk id ", chunk_id)

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
    def _llm_judge_chunk(self, proposition, candidate_chunk_ids) -> None | str:
        outline = ""
        for cid in candidate_chunk_ids:
            c = self.chunks[cid]
            outline += f"Chunk ID: {cid}\nSummary: {c['summary']}\n\n"

        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """
            Decide if the proposition belongs to any chunk below.
            Return ONLY the chunk_id or "No chunks".
            No explanation.
            No extra text.
            """),
            ("user", "Chunks:\n{outline}\nProposition:\n{proposition}")
        ])

        response = (PROMPT | self.llm | StrOutputParser()).invoke({
            "outline": outline,
            "proposition": proposition
        }).strip()

        return response if response in candidate_chunk_ids else None


    def _find_relevant_chunk(self, proposition)-> list[str] | None:
        prop_embedding = self.embedder.embed_query(proposition)
        scored_chunks = []
        for cid, chunk in self.chunks.items():
            # Uses the embedding of the Canonical Text now (much more accurate)
            score = self.cosine_similarity(prop_embedding, chunk['embedding'])
            scored_chunks.append((cid, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [cid for cid, score in scored_chunks[:5] if score > 0.75]

        if not top_candidates:
            print("No candidates found.")
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
