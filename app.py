from fastapi import FastAPI, APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import EmailStr
from typing import Annotated, Literal, Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, computed_field, AnyUrl, EmailStr
import requests

from utils.dataExtrationAndRendering import DataExtAndRenderingService

from config.config_file import Config


from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Multi Input Rag END-TO-END")



# rawDataDir = DataExtAndRenderingService.websiteDataExtration("https://parivahan.gov.in/")
rawDataDir = DataExtAndRenderingService.anyThingButJSOrSPA("https://parivahan.gov.in/")
print(rawDataDir)
Config.storeMDContent(rawDataDir)