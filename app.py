from docling.document_converter import DocumentConverter

source = "https://cittaai.com/"
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())