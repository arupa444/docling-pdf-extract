from fastapi import FastAPI, APIRouter, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import EmailStr
from typing import Annotated, Literal, Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, computed_field, AnyUrl, EmailStr
import requests
import shutil
import tempfile
from pathlib import Path
from rich import print
from urllib.parse import urlparse
import json
import os

from scrapy.crawler import CrawlerProcess
from utils.spider import FullWebsiteSpider

from utils.dataExtrationAndRendering import DataExtAndRenderingService

from config.config_file import Config

from utilsForRAG import agenticChunker, ragAnswer, DBretrieve, chunkMemoryIndex

from dotenv import load_dotenv

from pydantic import BaseModel

from fastapi import UploadFile, File, Form
import shutil
import tempfile
import os

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool



load_dotenv()
config = Config()
#
app = FastAPI(title="Multi Input Rag END-TO-END")
#
#
#
# rawDataDir = DataExtAndRenderingService.websiteDataExtration("https://parivahan.gov.in/")
# rawDataDir = DataExtAndRenderingService.anyThingButJSOrSPA("https://parivahan.gov.in/")
# print(rawDataDir)
# config.storeMDContent(rawDataDir)

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
# propositions = ac.process_accumulated_data(raw_text)
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
# config.save_results(propositions)
# #
# #

# OCR Part....

@app.post("/OCR_On_Single_Upload", summary="You can upload any kind of source file and get the OCR output....")
async def OCR_On_Single_Upload(
        file: UploadFile = File(...),
        subDir: str = Form('')
):

    file_suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        markdown_content = await DataExtAndRenderingService.anyThingButJSOrSPA(tmp_path)
        savedLocation = config.storeMDContent(markdown_content, subDir)
        return {"markdown_content": markdown_content, "SavedLocation": savedLocation}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/OCR_On_Folder_Or_Multiple_file_Uploads", summary="Upload a folder or select multiple files and collectively perform OCR and save all the results in a json......")
async def OCR_On_Folder_Or_Multiple_file_Upload(
        files: List[UploadFile] = File(...),
        subDir: str = Form('')
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
                markdown_content = await DataExtAndRenderingService.anyThingButJSOrSPA(file_path)

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

    config.jsonStoreForMultiDoc(results, subDir)
    return {"results": results}


@app.post("/OCR_On_nonJS_nonSPA_Website", summary="OCR on Non js and Non SPA website")
async def OCR_On_nonJS_nonSPA_Website(webLink: str = Form(...),
        subDir: str = Form('')):
    try:
        markdown_content = await DataExtAndRenderingService.anyThingButJSOrSPA(webLink)
        config.storeMDContent(markdown_content, subDir)
        return {"markdown_content": markdown_content}
    except Exception as e:
        return {"error": str(e)}


@app.post("/Multiple_OCRs_On_nonJS_nonSPA_Website", summary="Multiple OCRs on Non js and Non SPA website")
async def Multiple_OCRs_On_nonJS_nonSPA_Website(
        webLinks: List[str] = Form(...),
        subDir: str = Form('')
):

    cleaned_links = []
    for entry in webLinks:
        cleaned_links.extend([url.strip() for url in entry.split(',') if url.strip()])

    webLinks = cleaned_links

    results = []

    for webLink in webLinks:
        web_result = {}
        try:
            markdown_content = await DataExtAndRenderingService.anyThingButJSOrSPA(webLink)
            web_result = {
                "webName": webLink,
                "markdownContent": markdown_content,
                "status": "success",
            }

        except Exception as e:
            web_result = {
                "filename": webLink,
                "status": "error",
                "error": str(e)
            }

        results.append(web_result)

    config.jsonStoreForMultiDoc(results, subDir)
    return {"results": results}


@app.post("/OCR_On_JS_SPA_Website", summary="OCR on JS SPA website")
async def OCR_On_JS_SPA_Website(webLink: str = Form(...),
        subDir: str = Form('')):
    try:
        print(webLink)
        markdown_content = await DataExtAndRenderingService.websiteDataExtrationJs(webLink)
        config.storeMDContent(markdown_content, subDir)
        return {"markdown_content": markdown_content}
    except Exception as e:
        return {"error": str(e)}



@app.post("/Multiple_OCRs_On_JS_SPA_Websites", summary="Multiple OCRs on JS SPA website")
async def Multiple_OCRs_On_JS_SPA_Websites(
        webLinks: List[str] = Form(...),
        subDir: str = Form('')
):

    cleaned_links = []
    for entry in webLinks:
        cleaned_links.extend([url.strip() for url in entry.split(',') if url.strip()])

    webLinks = cleaned_links
    results = []

    for webLink in webLinks:
        web_result = {}
        try:
            markdown_content = await DataExtAndRenderingService.websiteDataExtrationJs(webLink)
            web_result = {
                "webName": webLink,
                "markdownContent": markdown_content,
                "status": "success",
            }

        except Exception as e:
            web_result = {
                "filename": webLink,
                "status": "error",
                "error": str(e)
            }

        results.append(web_result)

    config.jsonStoreForMultiDoc(results, subDir)
    return {"results": results}







# RAG Part....

@app.post("/RAG_On_Single_Upload", summary="You can upload any kind of source file and get the RAG output....")
async def RAG_On_Single_Upload(file: UploadFile = File(...), query: str = Form(...),
        subDir: str = Form('')):

    file_suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # 0
        markdown_content = await DataExtAndRenderingService.anyThingButJSOrSPA(tmp_path)
        savedLocation = config.storeMDContent(markdown_content, subDir)
        ac = agenticChunker.AgenticChunker()

        # 1. Raw Text Input
        raw_text = markdown_content

        # 2. Ingest Data (Layer 1)
        propositions = ac.process_accumulated_data(raw_text)

        print(f"\n[bold cyan]Generated {len(propositions)} Propositions[/bold cyan]")


        ac.add_propositions(propositions)
        ac.pretty_print_chunks()

        # 3. Build Memory Index (Layer 3)
        #    We initialize this AFTER ingestion is done.
        print("\n[bold blue]Building Memory Index...[/bold blue]")
        memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

        for chunk_id, chunk_data in ac.chunks.items():
            memory_index.add(chunk_id, chunk_data['embedding'])

        # 4. Retrieval (Layer 4)
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")

        # 5. RAG Answer (Layer 5)
        print("\n[bold blue]Generating Answer...[/bold blue]")


        final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)
        print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")



        config.save_results(savedLocation, propositions, ac.chunks, memory_index, subDir)
        return {"Top Result": f"{retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})" ,"Final Answer":final_answer, "markdown_content": markdown_content, "SavedLocation": savedLocation}


    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)










@app.post("/RAG_On_Folder_Or_Multiple_file_Uploads", summary="Upload a folder or select multiple files and collectively perform RAG and save all the results in a json......")
async def RAG_On_Folder_Or_Multiple_file_Uploads(
        files: List[UploadFile] = File(...),
        query: str = Form(...),
        subDir: str = Form('')
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
                markdown_content = await DataExtAndRenderingService.anyThingButJSOrSPA(file_path)

                file_result = {
                    "filename": file.filename,
                    "status": "success",
                    "markdown_content": markdown_content
                }

            except Exception as e:
                file_result = {
                    "filename": file.filename,
                    "status": "error",
                    "error while extracting data from multiple uploading files": str(e)
                }

            results.append(file_result)

            await file.close()
    try:
        savedLocation = config.jsonStoreForMultiDoc(results, subDir)
        ac = agenticChunker.AgenticChunker()
        # 1. Raw Text Input
        raw_text = results

        # 2. Ingest Data (Layer 1)
        propositions = ac.process_accumulated_data(raw_text)

        print(f"\n[bold cyan]Generated {len(propositions)} Propositions[/bold cyan]")

        ac.add_propositions(propositions)
        ac.pretty_print_chunks()

        # 3. Build Memory Index (Layer 3)
        #    We initialize this AFTER ingestion is done.
        print("\n[bold blue]Building Memory Index...[/bold blue]")
        memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

        for chunk_id, chunk_data in ac.chunks.items():
            memory_index.add(chunk_id, chunk_data['embedding'])

        # 4. Retrieval (Layer 4)
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")

        # 5. RAG Answer (Layer 5)
        print("\n[bold blue]Generating Answer...[/bold blue]")

        final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)
        print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")

        config.save_results(savedLocation, propositions, ac.chunks, memory_index, subDir)
        return {"Top Result": f"{retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})",
                "Final Answer": final_answer, "markdown_content": markdown_content, "SavedLocation": savedLocation}
    except Exception as e:
        return {"error while perform RAG... on multiple uploaded file...": str(e)}











@app.post("/RAG_On_nonJS_nonSPA_Website", summary="RAG on Non js and Non SPA website")
async def RAG_On_nonJS_nonSPA_Website(
        webLink: str = Form(...),
        query: str = Form(...),
        subDir: str = Form('')
):
    try:
        markdown_content = await DataExtAndRenderingService.anyThingButJSOrSPA(webLink)
        savedLocation = config.storeMDContent(markdown_content, subDir)
        ac = agenticChunker.AgenticChunker()

        # 1. Raw Text Input
        raw_text = markdown_content

        # 2. Ingest Data (Layer 1)
        propositions = ac.process_accumulated_data(raw_text)

        print(f"\n[bold cyan]Generated {len(propositions)} Propositions[/bold cyan]")

        ac.add_propositions(propositions)
        ac.pretty_print_chunks()

        # 3. Build Memory Index (Layer 3)
        #    We initialize this AFTER ingestion is done.
        print("\n[bold blue]Building Memory Index...[/bold blue]")
        memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

        for chunk_id, chunk_data in ac.chunks.items():
            memory_index.add(chunk_id, chunk_data['embedding'])

        # 4. Retrieval (Layer 4)
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")

        # 5. RAG Answer (Layer 5)
        print("\n[bold blue]Generating Answer...[/bold blue]")

        final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)
        print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")

        config.save_results(savedLocation, propositions, ac.chunks, memory_index, subDir)
        return {"Top Result": f"{retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})",
                "Final Answer": final_answer, "markdown_content": markdown_content, "SavedLocation": savedLocation}

    except Exception as e:
        return {"error": str(e)}



@app.post("/RAG_On_Multiple_nonJS_nonSPA_Website", summary="RAG on Multiple Non js and Non SPA website")
async def RAG_On_Multiple_nonJS_nonSPA_Website(
        webLinks: List[str] = Form(...),
        query: str = Form(...),
        subDir: str = Form('')
):
    markdown_content = ""

    cleaned_links = []
    for entry in webLinks:
        cleaned_links.extend([url.strip() for url in entry.split(',') if url.strip()])

    webLinks = cleaned_links

    results = []

    for webLink in webLinks:
        web_result = {}
        try:
            markdown_content = await DataExtAndRenderingService.anyThingButJSOrSPA(webLink)
            web_result = {
                "webName": webLink,
                "markdownContent": markdown_content,
                "status": "success",
            }

        except Exception as e:
            web_result = {
                "filename": webLink,
                "status": "error",
                "error": str(e)
            }

        results.append(web_result)
    try:
        savedLocation = config.jsonStoreForMultiDoc(results, subDir)
        ac = agenticChunker.AgenticChunker()
        # 1. Raw Text Input
        raw_text = results

        # 2. Ingest Data (Layer 1)
        propositions = ac.process_accumulated_data(raw_text)

        print(f"\n[bold cyan]Generated {len(propositions)} Propositions[/bold cyan]")

        ac.add_propositions(propositions)
        ac.pretty_print_chunks()

        # 3. Build Memory Index (Layer 3)
        #    We initialize this AFTER ingestion is done.
        print("\n[bold blue]Building Memory Index...[/bold blue]")
        memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

        for chunk_id, chunk_data in ac.chunks.items():
            memory_index.add(chunk_id, chunk_data['embedding'])

        # 4. Retrieval (Layer 4)
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")

        # 5. RAG Answer (Layer 5)
        print("\n[bold blue]Generating Answer...[/bold blue]")

        final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)
        print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")

        config.save_results(savedLocation, propositions, ac.chunks, memory_index, subDir)
        return {"Top Result": f"{retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})",
                "Final Answer": final_answer, "markdown_content": markdown_content, "SavedLocation": savedLocation}
    except Exception as e:
        return {"error while perform RAG... on multiple uploaded file...": str(e)}












@app.post("/RAG_On_JS_SPA_Website", summary="RAG on JS SPA website")
async def RAG_On_JS_SPA_Website(
        webLink: str = Form(...),
        query: str = Form(...),
        subDir: str = Form('')
):
    try:
        print(webLink)
        markdown_content = await DataExtAndRenderingService.websiteDataExtrationJs(webLink)
        savedLocation = config.storeMDContent(markdown_content, subDir)
        ac = agenticChunker.AgenticChunker()

        # 1. Raw Text Input
        raw_text = markdown_content

        # 2. Ingest Data (Layer 1)
        propositions = ac.process_accumulated_data(raw_text)

        print(f"\n[bold cyan]Generated {len(propositions)} Propositions[/bold cyan]")

        ac.add_propositions(propositions)
        ac.pretty_print_chunks()

        # 3. Build Memory Index (Layer 3)
        #    We initialize this AFTER ingestion is done.
        print("\n[bold blue]Building Memory Index...[/bold blue]")
        memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

        for chunk_id, chunk_data in ac.chunks.items():
            memory_index.add(chunk_id, chunk_data['embedding'])

        # 4. Retrieval (Layer 4)
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")

        # 5. RAG Answer (Layer 5)
        print("\n[bold blue]Generating Answer...[/bold blue]")

        final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)
        print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")

        config.save_results(savedLocation, propositions, ac.chunks, memory_index, subDir)
        return {"Top Result": f"{retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})",
                "Final Answer": final_answer, "markdown_content": markdown_content, "SavedLocation": savedLocation}

    except Exception as e:
        return {"error": str(e)}




@app.post("/RAG_On_Multiple_JS_SPA_Websites", summary="RAG on Multiple JS SPA website")
async def RAG_On_Multiple_JS_SPA_Websites(
        webLinks: List[str] = Form(...),
        query: str = Form(...),
        subDir: str = Form('')
):
    markdown_content = ""

    cleaned_links = []
    for entry in webLinks:
        cleaned_links.extend([url.strip() for url in entry.split(',') if url.strip()])

    webLinks = cleaned_links
    results = []

    for webLink in webLinks:
        web_result = {}
        try:
            markdown_content = await DataExtAndRenderingService.websiteDataExtrationJs(webLink)
            web_result = {
                "webName": webLink,
                "markdownContent": markdown_content,
                "status": "success",
            }

        except Exception as e:
            web_result = {
                "filename": webLink,
                "status": "error",
                "error": str(e)
            }

        results.append(web_result)

    try:
        savedLocation = config.jsonStoreForMultiDoc(results, subDir)
        ac = agenticChunker.AgenticChunker()
        # 1. Raw Text Input
        raw_text = results

        # 2. Ingest Data (Layer 1)
        propositions = ac.process_accumulated_data(raw_text)

        print(f"\n[bold cyan]Generated {len(propositions)} Propositions[/bold cyan]")

        ac.add_propositions(propositions)
        ac.pretty_print_chunks()

        # 3. Build Memory Index (Layer 3)
        #    We initialize this AFTER ingestion is done.
        print("\n[bold blue]Building Memory Index...[/bold blue]")
        memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

        for chunk_id, chunk_data in ac.chunks.items():
            memory_index.add(chunk_id, chunk_data['embedding'])

        # 4. Retrieval (Layer 4)
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")

        # 5. RAG Answer (Layer 5)
        print("\n[bold blue]Generating Answer...[/bold blue]")

        final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)
        print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")

        config.save_results(savedLocation, propositions, ac.chunks, memory_index, subDir)
        return {"Top Result": f"{retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})",
                "Final Answer": final_answer, "markdown_content": markdown_content, "SavedLocation": savedLocation}
    except Exception as e:
        return {"error while perform RAG... on multiple uploaded file...": str(e)}



@app.post("/RAG_From_Uploaded_Index", summary="Upload .faiss, .json (ids), and .json (chunks) to chat")
async def rag_from_uploaded_index(
        query: str = Form(...),
        file_faiss: UploadFile = File(..., description="The .faiss index file"),
        file_ids: UploadFile = File(..., description="The _ids.json file"),
        file_chunks: UploadFile = File(..., description="The _chunks.json file")
):
    temp_dir = tempfile.mkdtemp()

    try:
        # 1. Save Uploaded Files to Temp
        path_faiss = os.path.join(temp_dir, file_faiss.filename)
        path_ids = os.path.join(temp_dir, file_ids.filename)
        path_chunks = os.path.join(temp_dir, file_chunks.filename)

        with open(path_faiss, "wb") as f:
            shutil.copyfileobj(file_faiss.file, f)
        with open(path_ids, "wb") as f:
            shutil.copyfileobj(file_ids.file, f)
        with open(path_chunks, "wb") as f:
            shutil.copyfileobj(file_chunks.file, f)

        # 2. Initialize Helper Classes
        ac = agenticChunker.AgenticChunker()
        memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

        # 3. Load Data
        print("[INFO] Loading Chunks...")
        with open(path_chunks, "r", encoding="utf-8") as f:
            ac.chunks = json.load(f)

        print("[INFO] Loading Index...")
        memory_index.load_local(path_faiss, path_ids)

        # 4. Retrieve
        print(f"[INFO] Searching for: {query}")
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        if not retrieved_docs:
            return {"message": "No matches found.", "score": 0}

        # 5. Answer
        print("[INFO] Generating Answer...")
        final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)

        return {
            "query": query,
            "final_answer": final_answer,
            "top_source": retrieved_docs[0]['title'],
            "score": retrieved_docs[0]['score']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Cleanup Temp Files
        shutil.rmtree(temp_dir)



from utilsForRAG.ConversationManager import ConversationManager
from utils.helper_file import HelperFile

helperFile = HelperFile()

# Global instance (faster than recreating)
conv_manager = ConversationManager()

# Cache for loaded files (avoid reloading same files)
file_cache = {}


@app.post("/RAG_Conversational_Endpoint", summary="Optimized RAG with 500-token context")
async def rag_conversational_endPoint(
        session_id: str = Form(...),
        query: str = Form(...),
        file_faiss: UploadFile = File(...),
        file_ids: UploadFile = File(...),
        file_chunks: UploadFile = File(...)
):
    try:
        # Read file contents once
        faiss_content = await file_faiss.read()
        ids_content = await file_ids.read()
        chunks_content = await file_chunks.read()

        # Generate cache keys
        cache_key = f"{helperFile.get_file_hash(faiss_content)}_{helperFile.get_file_hash(chunks_content)}"

        # Check cache first (HUGE speedup for repeated queries)
        if cache_key in file_cache:
            print("[CACHE HIT] Using cached index")
            ac = file_cache[cache_key]["chunker"]
            memory_index = file_cache[cache_key]["index"]
        else:
            print("[CACHE MISS] Loading files")
            temp_dir = tempfile.mkdtemp()

            try:
                # Save files
                path_faiss = os.path.join(temp_dir, "index.faiss")
                path_ids = os.path.join(temp_dir, "ids.json")
                path_chunks = os.path.join(temp_dir, "chunks.json")

                with open(path_faiss, "wb") as f:
                    f.write(faiss_content)
                with open(path_ids, "wb") as f:
                    f.write(ids_content)
                with open(path_chunks, "wb") as f:
                    f.write(chunks_content)

                # Initialize
                ac = agenticChunker.AgenticChunker()
                memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

                # Load
                with open(path_chunks, "r", encoding="utf-8") as f:
                    ac.chunks = json.load(f)
                memory_index.load_local(path_faiss, path_ids)

                # Cache it (limit cache size)
                if len(file_cache) > 10:
                    file_cache.clear()  # Simple eviction
                file_cache[cache_key] = {
                    "chunker": ac,
                    "index": memory_index
                }

            finally:
                shutil.rmtree(temp_dir)

        # Get context (fast)
        context_str = conv_manager.get_context(session_id)

        # Retrieve documents
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        if not retrieved_docs:
            error_msg = "No matches found in the knowledge base."
            conv_manager.add_exchange(session_id, query, error_msg)
            return {
                "session_id": session_id,
                "message": error_msg,
                "score": 0,
                "cached": cache_key in file_cache
            }

        # Build evidence (only top 3 for speed)
        evidence_text = "\n\n".join(
            f"SOURCE: {c['chunk_id']}\n" + "\n".join(f"- {p}" for p in c["evidence"][:5])
            for c in retrieved_docs[:3]  # Limit to top 3 chunks
        )

        # Generate answer with minimal prompt
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        CONV_PROMPT = ChatPromptTemplate.from_messages([
            ("system", "Answer using conversation context and evidence. Cite sources. Be concise."),
            ("user", "{context}Q: {query}\n\nEvidence:\n{evidence}")
        ])

        runnable = CONV_PROMPT | ac.llm | StrOutputParser()
        final_answer = runnable.invoke({
            "context": context_str,
            "query": query,
            "evidence": evidence_text
        })

        # Update history (efficient)
        conv_manager.add_exchange(session_id, query, final_answer)

        # Get stats
        stats = conv_manager.get_stats(session_id)

        return {
            "session_id": session_id,
            "query": query,
            "final_answer": final_answer,
            "top_source": retrieved_docs[0]['title'],
            "score": float(retrieved_docs[0]['score']),
            "context_tokens": stats["token_count"],
            "message_count": stats["message_count"],
            "cached": True  # File was cached
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/clear_session_optimized")
async def clear_session_optimized(session_id: str = Form(...)):
    """Clear session history"""
    conv_manager.clear(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.get("/session_stats")
async def session_stats(session_id: str):
    """Get session statistics"""
    return conv_manager.get_stats(session_id)


@app.post("/clear_file_cache")
async def clear_file_cache():
    """Clear the file cache (admin use)"""
    file_cache.clear()
    return {"message": "File cache cleared"}



@app.post("/full_website_extraction")
async def full_website_extraction(
        # background_tasks: BackgroundTasks,
        webSite: str = Form(...)
):
    try:
        helperFile.run_spider_process(webSite)
        print("Full website extraction complete")
        return {"message": "Full website extraction complete"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/full_website_extraction_and_conversation")
async def full_website_extraction_and_conversation(
        # background_tasks: BackgroundTasks,
        webSite: str = Form(...)
):
    try:
        helperFile.run_spider_process(webSite)
        print("Full website extraction complete")
        allowed_domain = urlparse(webSite).netloc
        print("the dir name : ", allowed_domain)

        raw_text = helperFile.run_HTMLs_PDFs_to_MDFile_process(allowed_domain)
        return {"message": "Full website extraction and conversation complete",
                "raw_text": raw_text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/full_website_extraction_conversation_and_execution")
async def full_website_extraction_conversation_and_execution(
        # background_tasks: BackgroundTasks,
        query: str = Form(...),
        webSite: str = Form(...)
):
    try:
        helperFile.run_spider_process(webSite)
        print("Full website extraction complete")
        allowed_domain = urlparse(webSite).netloc
        print("the dir name : ", allowed_domain)

        raw_text = helperFile.run_HTMLs_PDFs_to_MDFile_process(allowed_domain)

        # print("\n\n\n\n\n\n\n\n\n\n\n\n\n",type(raw_text),"\n\n\n\n\n\n\n\n\n\n\n\n\n","\n\n\n\n\n\n\n\n\n\n\n\n\n")
        #
        # print("\n\n\n\n\n\n\n\n\n\n\n\n\n",raw_text,"\n\n\n\n\n\n\n\n\n\n\n\n\n")

        print("HTMLs PDFs to MDFile process complete.....")
        ac = agenticChunker.AgenticChunker()

        propositions = ac.process_accumulated_data(raw_text)


        # only and only if you think, it will stop inbetween


        extension = ".json"
        fileNameForPropositions = f"Citta_Propositions_{allowed_domain}{extension}"

        print(f"\n[bold cyan]Generated {len(propositions)} Propositions[/bold cyan]")


        prop_path = os.path.join("vectorStoreDB", fileNameForPropositions)
        config.makeDirectories(os.path.dirname(prop_path))
        with open(prop_path, "w", encoding="utf-8") as f:
            json.dump(propositions, f, indent=4, ensure_ascii=False)


        ac.add_propositions(propositions)
        ac.pretty_print_chunks()

        # 3. Build Memory Index (Layer 3)
        #    We initialize this AFTER ingestion is done.
        print("\n[bold blue]Building Memory Index...[/bold blue]")
        memory_index = chunkMemoryIndex.ChunkMemoryIndex(dim=768)

        for chunk_id, chunk_data in ac.chunks.items():
            memory_index.add(chunk_id, chunk_data['embedding'])

        # 4. Retrieval (Layer 4)
        retrieved_docs = DBretrieve.Retrieve.retrieve(query, ac, memory_index)

        print(f"\n[green]Top Result:[/green] {retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})")

        # 5. RAG Answer (Layer 5)
        print("\n[bold blue]Generating Answer...[/bold blue]")

        final_answer = ragAnswer.Answer.answer(query, retrieved_docs, ac.llm)
        print(f"\n[bold]Final Answer:[/bold]\n{final_answer}")

        config.save_results(allowed_domain, propositions, ac.chunks, memory_index, allowed_domain)
        return {"Top Result": f"{retrieved_docs[0]['title']} (Score: {retrieved_docs[0]['score']:.4f})",
                "Final Answer": final_answer, "SavedLocation": allowed_domain}
    except Exception as e:
        return {"error while perform RAG... on multiple uploaded file...": str(e)}
