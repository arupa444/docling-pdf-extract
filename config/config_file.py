import os
import json
from datetime import datetime
from typing import List
from rich import print
import sqlite3

class Config:
    # this is because my python is crasing out.....
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    @staticmethod
    def makeDirectories(dirName: str) -> None:
        os.makedirs(dirName, exist_ok=True)


    @staticmethod
    def storeMDContent(rawData: str, subDir: str = "", target_dir: str = "rawDataDir") -> str | None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_dir = f"{target_dir}/{subDir}_{timestamp}"
        Config.makeDirectories(target_dir)
        extension = ".md"
        filename = f"Citta_{timestamp}{extension}"
        full_path = os.path.join(target_dir, filename)
        try:
            with open(full_path, "w", encoding="utf-8") as file:
                file.write(rawData)
            print(f"File successfully saved at: {full_path}")
            return str(filename)
        except Exception as e:
            print(f"Error saving file: {e}")

    @staticmethod
    def save_results(savedLocation: str, propositions: list, chunks: dict, memory_index, subDir: str = "",
                     folder_name: str = "vectorStoreDB"):

        extension = ".json"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        savedLocation = savedLocation.split(".")[0]
        folder_name = f"{folder_name}/{subDir}_{timestamp}"
        Config.makeDirectories(folder_name)

        # File Names
        fileNameForPropositions = f"Citta_Propositions_{savedLocation}{extension}"
        fileNameForChunks = f"Citta_Chunks_{savedLocation}{extension}"
        extension1 = ".db"
        fileNameForChunksDB = f"Citta_Chunks_{savedLocation}{extension1}"

        # Create Folder
        os.makedirs(folder_name, exist_ok=True)

        print("Folder successfully created")


        conn = sqlite3.connect(fileNameForChunksDB)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            chunk_index INTEGER,
            title TEXT,
            summary TEXT,
            propositions TEXT,
            canonical_text TEXT
        )
        """)

        inserted = 0

        for _, chunk in chunks.items():
            cur.execute("""
                INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk["chunk_id"],
                chunk.get("chunk_index"),
                chunk.get("title"),
                chunk.get("summary"),
                json.dumps(chunk.get("propositions", [])),
                chunk.get("canonical_text")
            ))
            inserted += 1

        conn.commit()
        conn.close()

        print(f"{inserted} chunks inserted into chunks.db")


        # 1. Save Propositions
        prop_path = os.path.join(folder_name, fileNameForPropositions)
        with open(prop_path, "w", encoding="utf-8") as f:
            json.dump(propositions, f, indent=4, ensure_ascii=False)

        # 2. Save Chunks (Text Data)
        chunk_path = os.path.join(folder_name, fileNameForChunks)
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)

        # 3. Save FAISS Index & IDs (The new part)
        # Using the same savedLocation ID for the filename
        index_prefix = f"Citta_Index_{savedLocation}"
        memory_index.save_local(folder_name, index_prefix)

        print(f"[SUCCESS] All data saved in {folder_name} with ID: {savedLocation}")


    @staticmethod
    def jsonStoreForMultiDoc(rawData: list, subDir: str = "", target_dir: str = "rawDataDir") -> str | None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_dir = f"{target_dir}/{subDir}_{timestamp}"
        Config.makeDirectories(target_dir)
        extension = ".json"
        filename = f"Citta_{timestamp}{extension}"
        full_path = os.path.join(target_dir, filename)
        try:
            with open(full_path, "w", encoding="utf-8") as file:
                json.dump(rawData, file, indent=4, ensure_ascii=False)
            print(f"File successfully saved at: {full_path}")
            return str(filename)
        except Exception as e:
            print(f"Error saving file: {e}")