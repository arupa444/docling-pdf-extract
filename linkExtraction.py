from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from playwright.sync_api import sync_playwright

source = "https://arxiv.org/pdf/2408.09869"


def get_dynamic_html(url):
    """Fetches the fully rendered HTML using a headless browser."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)

        # Wait for the page to load (you can adjust this or wait for a specific element)
        page.wait_for_timeout(5000)  # waiting 5 seconds for JS to render

        html_content = page.content()
        browser.close()
        return html_content


# 1. Get the rendered HTML string
rendered_html = get_dynamic_html(source)

# 2. Convert the string using Docling
converter = DocumentConverter()
# We use 'convert_string' instead of 'convert' since we have the HTML as text now
result = converter.convert_string(rendered_html, InputFormat.HTML)

print(result.document.export_to_markdown())