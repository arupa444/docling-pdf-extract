from playwright.async_api import async_playwright

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