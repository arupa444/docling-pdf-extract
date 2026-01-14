from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import os
from helper_file import HelperFile



class DataExtrationAndRenderingService:
    os.makedirs("images", exist_ok=True)

    @staticmethod
    def anyThingButJSOrSPA(source: str) -> str:
        converter = DocumentConverter()
        result = converter.convert(source)
        return result.document.export_to_markdown()

    @staticmethod
    def websiteDataExtration(source: str) -> str:
        rendered_html = HelperFile.get_dynamic_html(source)
        converter = DocumentConverter()
        result = converter.convert_string(rendered_html, InputFormat.HTML)
        return result.document.export_to_markdown()
