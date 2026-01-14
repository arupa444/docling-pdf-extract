from playwright.sync_api import sync_playwright

class HelperFile:

    @staticmethod
    def get_dynamic_html(url: str):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)
            page.wait_for_timeout(15000)
            html_content = page.content()
            browser.close()
            return html_content