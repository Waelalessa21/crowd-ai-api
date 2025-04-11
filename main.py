import os
import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from predictor import predict_density
from utils import render_density_map

app = FastAPI(title="Crowd Counter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crowd Counter API"}

@app.post("/predict")
async def predict_crowd(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    count, density_map = predict_density(image)
    heatmap = render_density_map(density_map)
    
    buffered_original = io.BytesIO()
    image.save(buffered_original, format="JPEG")
    original_base64 = base64.b64encode(buffered_original.getvalue()).decode('utf-8')
    
    buffered_heatmap = io.BytesIO()
    heatmap.save(buffered_heatmap, format="PNG")
    heatmap_base64 = base64.b64encode(buffered_heatmap.getvalue()).decode('utf-8')
    
    return {
        "count": count,
        "original_image": original_base64,
        "heatmap_image": heatmap_base64
    }