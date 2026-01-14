import os
from datetime import datetime

class Config:

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
            return full_path
        except Exception as e:
            print(f"Error saving file: {e}")