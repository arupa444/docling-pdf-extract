import sys
import threading
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
allowed_domain = sys.argv[1]
BASE_DIR = Path(f"storeCurlData/{allowed_domain}")
PDF_DIR = BASE_DIR / "files"
HTML_DIR = BASE_DIR / "html"
OUTPUT_DIR = Path(f"markdown_output/{allowed_domain}")

# A lock to keep console output clean (so lines don't mix up)
print_lock = threading.Lock()

# ---------------------------------------------------------
# 2. INITIALIZE CONVERTER (Thread-Safe Loading)
# ---------------------------------------------------------
# We initialize one global converter.
# Note: Docling/PyTorch usually handles threading okay, but if you run
# into memory issues, the script might slow down as they fight for CPU.
print("Initializing Docling Converter...")

pipeline_options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend
        )
    }
)


# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def thread_safe_print(message):
    """Prevents messy console output when threads run together."""
    with print_lock:
        print(message)


def save_to_file(content: str, original_path: Path, subfolder_name: str):
    target_folder = OUTPUT_DIR / subfolder_name
    target_folder.mkdir(parents=True, exist_ok=True)

    new_filename = original_path.with_suffix('.md').name
    save_path = target_folder / new_filename

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    thread_safe_print(f"  [{subfolder_name.upper()} DONE] Saved: {new_filename}")


# ---------------------------------------------------------
# 4. THREAD WORKERS
# ---------------------------------------------------------

def task_process_pdfs():
    """Worker function for the PDF thread."""
    if not PDF_DIR.exists():
        thread_safe_print(f"Warning: PDF directory not found at {PDF_DIR}")
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    thread_safe_print(f"\n--- [PDF Thread] Found {len(pdf_files)} PDF files ---")

    for file_path in pdf_files:
        try:
            thread_safe_print(f"[PDF START] {file_path.name}...")

            # The Heavy Lifting
            result = converter.convert(file_path)
            markdown = result.document.export_to_markdown()

            save_to_file(markdown, file_path, "files")

        except Exception as e:
            thread_safe_print(f"  [PDF ERROR] Failed {file_path.name}: {e}")

    thread_safe_print("\n--- [PDF Thread] Finished ---")


def task_process_htmls():
    """Worker function for the HTML thread."""
    if not HTML_DIR.exists():
        thread_safe_print(f"Warning: HTML directory not found at {HTML_DIR}")
        return

    html_files = list(HTML_DIR.glob("*.html"))
    thread_safe_print(f"\n--- [HTML Thread] Found {len(html_files)} HTML files ---")

    for file_path in html_files:
        try:
            thread_safe_print(f"[HTML START] {file_path.name}...")

            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            result = converter.convert_string(html_content, InputFormat.HTML)
            markdown = result.document.export_to_markdown()

            save_to_file(markdown, file_path, "html")

        except Exception as e:
            thread_safe_print(f"  [HTML ERROR] Failed {file_path.name}: {e}")

    thread_safe_print("\n--- [HTML Thread] Finished ---")


# ---------------------------------------------------------
# 5. MAIN EXECUTION (PARALLEL)
# ---------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create the two threads
    pdf_thread = threading.Thread(target=task_process_pdfs, name="PDF-Worker")
    html_thread = threading.Thread(target=task_process_htmls, name="HTML-Worker")

    print("--- Starting Parallel Processing ---")

    # Start them
    html_thread.start()  # HTMLs are fast, start them
    pdf_thread.start()  # PDFs are slow, start them

    # Wait for both to finish before exiting the script
    html_thread.join()
    pdf_thread.join()

    print("\n--- All Threads Completed Successfully ---")