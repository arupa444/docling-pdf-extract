from playwright.async_api import async_playwright
import subprocess
import sys

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
        # This runs the separate python script
        # sys.executable ensures we use the same python environment (venv)
        subprocess.run([sys.executable, "HTMLs_PDFs_to_MD.py", subDirName])