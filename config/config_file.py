import os
import json
from datetime import datetime
from typing import List
from rich import print

class Config:
    # this is because my python is crasing out.....
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    @staticmethod
    def makeDirectories(dirName: str) -> None:
        os.makedirs(dirName, exist_ok=True)


    @staticmethod
    def storeMDContent(rawData: str, target_dir: str = "rawDataDir") -> str | None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        extension = ".md"
        filename = f"Citta_{timestamp}{extension}"
        full_path = os.path.join(target_dir, filename)
        try:
            with open(full_path, "w", encoding="utf-8") as file:
                file.write(rawData)
            print(f"File successfully saved at: {full_path}")
            return str(timestamp)
        except Exception as e:
            print(f"Error saving file: {e}")

    @staticmethod
    def save_results(timestamp: str, propositions: List[str], chunks: dict, folder_name: str = "vectorStoreDB"):
        extension = ".json"

        fileNameForPropositions = f"Citta_Propositions_{timestamp}{extension}"
        fileNameForChunks = f"Citta_Chunks_{timestamp}{extension}"


        os.makedirs(folder_name, exist_ok=True)

        # Save Propositions to JSON
        prop_path = os.path.join(folder_name, fileNameForPropositions)
        with open(prop_path, "w", encoding="utf-8") as f:
            json.dump(propositions, f, indent=4, ensure_ascii=False)
        print(f"Saved propositions to: [underline]{prop_path}[/underline]")

        # 3. Save Chunks to JSON
        chunk_path = os.path.join(folder_name, fileNameForChunks)
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        print(f"Saved chunks to: [underline]{chunk_path}[/underline]")
