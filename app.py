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


@app.post("/OCR_On_Single_Upload", summary="You can upload any kind of source file and get the OCR output....")
async def OCR_On_Single_Upload(file: UploadFile = File(...)):

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


@app.post("/OCR_On_Folder_Or_Multiple_file_Uploads", summary="Upload a folder or select multiple files and collectively perform OCR and save all the results in a json......")
async def OCR_On_Folder_Or_Multiple_file_Upload(
        files: List[UploadFile] = File(...)
):
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:

        for file in files:
            file_result = {}
            file_path = None

            try:
                safe_filename = Path(file.filename).name
                file_path = os.path.join(temp_dir, safe_filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                markdown_content = DataExtAndRenderingService.anyThingButJSOrSPA(file_path)

                file_result = {
                    "filename": file.filename,
                    "status": "success",
                    "markdown_content": markdown_content
                }

            except Exception as e:
                file_result = {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                }

            results.append(file_result)

            await file.close()

    config.jsonStoreForMultiDoc(results)
    return {"results": results}










@app.post("/OCR_On_nonJS_nonSPA_Website", summary="You can upload any kind of source file")
async def OCR_On_nonJS_nonSPA_Website(file: UploadFile = File(...)):

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


@app.post("/Multiple_OCRs_On_nonJS_nonSPA_Website", summary="Upload a folder (select multiple files)")
async def Multiple_OCRs_On_nonJS_nonSPA_Website(
        files: List[UploadFile] = File(...)
):
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:

        for file in files:
            file_result = {}
            file_path = None

            try:
                safe_filename = Path(file.filename).name
                file_path = os.path.join(temp_dir, safe_filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                markdown_content = DataExtAndRenderingService.anyThingButJSOrSPA(file_path)



                file_result = {
                    "filename": file.filename,
                    "markdown_content": markdown_content,
                    "status": "success",
                }

            except Exception as e:
                file_result = {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                }

            results.append(file_result)
            await file.close()

    config.jsonStoreForMultiDoc(results)
    return {"results": results}
















@app.post("/OCR_On_JS_SPA_Website", summary="You can upload any kind of source file")
async def OCR_On_JS_SPA_Website(file: UploadFile = File(...)):

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



@app.post("/Multiple_OCRs_On_JS_SPA_Websites", summary="Upload a folder (select multiple files)")
async def Multiple_OCRs_On_JS_SPA_Websites(
        files: List[UploadFile] = File(...)
):
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:

        for file in files:
            file_result = {}
            file_path = None

            try:
                safe_filename = Path(file.filename).name
                file_path = os.path.join(temp_dir, safe_filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                markdown_content = DataExtAndRenderingService.anyThingButJSOrSPA(file_path)



                file_result = {
                    "filename": file.filename,
                    "status": "success",
                    "markdown_content": markdown_content
                }

            except Exception as e:
                file_result = {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                }

            results.append(file_result)
            await file.close()

    config.jsonStoreForMultiDoc(results)
    return {"results": results}
