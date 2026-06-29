from fastapi import FastAPI, File, UploadFile, HTTPException
from app.face_occlusion import Occlusion
from app.face_match import Matching
from typing import List, Annotated
import os
from pydantic import Field

app = FastAPI()
occlusion = Occlusion()
face_match = Matching()

VALID_FORMAT = {"image/jpeg", "image/png", "image/jpg"}

@app.get("/")
async def root():
    return { "status": "SUCCESS"}

@app.post("/match")
async def create_upload_files(files: Annotated[List[UploadFile], File(), Field(min_length=2, max_length=2)]):
    file_paths = []
    
    os.makedirs("./imgs", exist_ok=True)
    for file in files:
        if file.content_type not in VALID_FORMAT:
            raise HTTPException(status_code=400, detail="Invalid image format")
        content = await file.read()
        file_path = os.path.join("app/imgs", file.filename)
        file_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(content)

    occluded1 = occlusion.predict(file_paths[0])
    occluded2 = occlusion.predict(file_paths[1])
    if occluded1["status"] != "SUCCESS":
        return occluded1
    if occluded2["status"] != "SUCCESS":
        return occluded2
    return face_match.match(file_paths[0], file_paths[1], 0.7)
