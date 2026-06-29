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

@app.post("/upload")
async def create_upload_files(files: Annotated[List[UploadFile], File(), Field(min_length=2, max_length=2)]):
    results = []
    file_paths = []
    
    for file in files:
        if file.content_type not in VALID_FORMAT:
            raise HTTPException(status_code=400, detail="Invalid image format")
        content = await file.read()
        file_path = os.path.join("imgs", file.filename)
        file_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(content)
        results.append({"filename": file.filename, "size": len(content)})

    occluded1 = occlusion.predict(file_paths[0])
    occluded2 = occlusion.predict(file_paths[1])
    if occluded1["status"] != "SUCCESS":
        return occluded1
    if occluded2["status"] != "SUCCESS":
        return occluded2
    if occluded1["occluded"] or occluded2["occluded"]:
        return {"status": "FACE_OCCLUDED", 
                file_paths[0]: "OCCLUDED" if occluded1["occluded"] else "NOT_OCCLUDED",
                file_paths[1]: "OCCLUDED" if occluded2["occluded"] else "NOT_OCCLUDED",
                }
    if not occluded1["occluded"] and not occluded2["occluded"]:
        return face_match.match(file_paths[0], file_paths[1], 0.7)
