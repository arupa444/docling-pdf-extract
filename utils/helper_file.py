from playwright.async_api import async_playwright
import subprocess
import sys
from pathlib import Path
import json

class HelperFile:
    @staticmethod
    async def get_dynamic_html(url: str):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)
            await page.wait_for_timeout(15000)
            html_content = await page.content()
            await browser.close()
            return html_content

    @staticmethod
    def get_file_hash(file_content: bytes) -> str:
        """Fast hash for file caching"""
        import hashlib
        return hashlib.md5(file_content).hexdigest()[:16]

    @staticmethod
    def run_spider_process(url: str):
        # This runs the separate python script
        # sys.executable ensures we use the same python environment (venv)
        subprocess.run([sys.executable, "run_spidy.py", url])

    @staticmethod
    def run_HTMLs_PDFs_to_MDFile_process(subDirName: str):

        subprocess.run([sys.executable, "HTMLs_PDFs_to_MD.py", subDirName], check=True)
        json_output_path = Path(f"storeCurlData/{subDirName}/{subDirName}.json")

        # 3. Read the file and return the data
        if json_output_path.exists():
            try:
                with open(json_output_path, "r", encoding="utf-8") as f:
                    accumulated_results = json.load(f)
                return accumulated_results
            except json.JSONDecodeError:
                print("Error: The generated JSON file was corrupted.")
                return []
        else:
            print(f"Error: Expected output file not found at {json_output_path}")
            return []