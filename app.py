from fastapi import FastAPI, APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import EmailStr
from typing import Annotated, Literal, Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, computed_field, AnyUrl, EmailStr
import requests
import shutil
import tempfile
from pathlib import Path
import os

from utils.dataExtrationAndRendering import DataExtAndRenderingService

from config.config_file import Config

from utilsForRAG import agenticChunker, ragAnswer, DBretrieve, chunkMemoryIndex

from dotenv import load_dotenv
load_dotenv()
config = Config()
#
app = FastAPI(title="Multi Input Rag END-TO-END")
#
#
#
# # rawDataDir = DataExtAndRenderingService.websiteDataExtration("https://parivahan.gov.in/")
# # rawDataDir = DataExtAndRenderingService.anyThingButJSOrSPA("https://parivahan.gov.in/")
# # print(rawDataDir)
# # config.storeMDContent(rawDataDir)
#
# ac = agenticChunker.AgenticChunker()
#
# # 1. Raw Text Input
# raw_text = """
# The Apollo program was a series of spaceflight missions conducted by NASA between 1961 and 1972.
# It succeeded in landing the first humans on the Moon in 1969.
# Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above.
# Meanwhile, in the ocean depths, the blue whale is the largest animal known to have ever lived.
# It can reach lengths of up to 29.9 meters and weigh 199 metric tons.
# Blue whales feed almost exclusively on krill.
# """
#
# # 2. Ingest Data (Layer 1)
# propositions = ac.generate_propositions(raw_text)
# ac.add_propositions(propositions)
# ac.pretty_print_chunks()
#
# # 3. Build Memory Index (Layer 3)
# #    We initialize this AFTER ingestion is done.
# print("\n[bold blue]Building Memory Index...[/bold blue]")
# memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)
#
# for chunk_id, chunk_data in ac.chunks.items():
#     memory_index.add(chunk_id, chunk_data['embedding'])
#
# # 4. Retrieval (Layer 4)
# query = "Who walked on the moon?"
# retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)
#
# print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")
#
# # 5. RAG Answer (Layer 5)
# print("\n[bold blue]Generating Answer...[/bold blue]")
# final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)
# print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")
# ac.save_results(propositions)
#
#


@app.post("/OCR_On_Single_Upload", summary="You can upload any kind of source file")
async def single_upload(file: UploadFile = File(...)):

    file_suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        markdown_content = DataExtAndRenderingService.anyThingButJSOrSPA(tmp_path)
        config.storeMDContent(markdown_content)
        return {"markdown_content": markdown_content}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ... (Assume your DataExtAndRenderingService and imports are here) ...

@app.post("/OCR_On_Folder_Upload", summary="Upload a folder (select multiple files)")
async def folder_upload(
        files: List[UploadFile] = File(...)
):
    results = []

    # Create a temporary directory to hold this batch of files
    # This is cleaner than creating individual temp files in the global temp dir
    with tempfile.TemporaryDirectory() as temp_dir:

        for file in files:
            file_result = {}
            file_path = None

            try:
                # 1. Construct the path inside our temp directory
                # We use the filename to keep extensions correct for Docling
                safe_filename = Path(file.filename).name
                file_path = os.path.join(temp_dir, safe_filename)

                # 2. Save the uploaded file to disk
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # 3. Process the file
                # We reuse your existing service logic
                markdown_content = DataExtAndRenderingService.anyThingButJSOrSPA(file_path)

                # 4. Store/Save logic
                config.storeMDContent(markdown_content)

                file_result = {
                    "filename": file.filename,
                    "status": "success",
                    "markdown_content": markdown_content
                }

            except Exception as e:
                # If one file fails, we log the error but continue processing the others
                file_result = {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                }

            results.append(file_result)

            # Explicitly close the file handle from FastAPI
            await file.close()

    # The TemporaryDirectory is automatically deleted here when we exit the 'with' block
    return {"results": results}