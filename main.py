import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, File, UploadFile, status, Header, Request, Depends, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from typing import List
from tempfile import NamedTemporaryFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import string
import random
import photos
import os
import ai
import io


app = FastAPI()

baseURL = 'http://localhost:8000/models'

app.mount("/models", StaticFiles(directory="models"), name="models")
app.mount("/photos", StaticFiles(directory="photos"), name="photos")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_html_page():
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/photo")
async def getUser(item: UploadFile = File(...)):
    res = ''.join(random.choices(string.ascii_uppercase +
                             string.digits, k=10))
    with open(f"photos/{res}.jpg", "wb") as f:
        f.write(await item.read())
    return {"body": f"{baseURL}/avatar.glb",
            "hair": f"{baseURL}/eyes.glb",
            "eye": f"{baseURL}/hair.glb"}

@app.post("/photos")
async def getUser(email, photo1: UploadFile = File(...), photo2: UploadFile = File(...), photo3: UploadFile = File(...), photo4: UploadFile = File(...)):
    photoss = [photo1,photo2,photo3,photo4]
    if len(photoss) != 4:
        raise HTTPException(status_code=400, detail="Необходимо загрузить 4 фотографии")
    saved_photos = []
    for photo in photoss:
        saved_location = await photos.save_user_photo(email, photo)
        saved_photos.append(saved_location)
    
    return {"message": "Фотографии успешно загружены", "saved_locations": saved_photos}

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Сохраняем загруженный файл временно
        with open("test.jpg", 'wb') as temp:
            temp.write(file.file.read())

        # Извлекаем лицо из изображения
        extracted_face = ai.extract_face('test.jpg')

        if isinstance(extracted_face, bytes):
            # Отправляем изображение в ответе
            return StreamingResponse(io.BytesIO(extracted_face), media_type="image/jpeg")

        # Если лицо не было найдено
        elif isinstance(extracted_face, str):
            raise HTTPException(status_code=400, detail=extracted_face)

        # Если возникла другая ошибка
        else:
            raise HTTPException(status_code=500, detail="Ошибка обработки изображения")


    except Exception as e:
        # В случае любых ошибок возвращаем 500 с описанием ошибки
        raise HTTPException(status_code=500, detail=str(e))

