import io
import time
import cv2
import numpy as np
from fastapi import FastAPI, APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
from src.config import TaiyoConfig
from src.handler import TaiyoTcapture
from src.utils.logger import Logger
from typing import List, Tuple

api_router = APIRouter()
tcapture_handler = TaiyoTcapture(config=TaiyoConfig)
logger = Logger(config=TaiyoConfig)

async def decode_image(uploaded_file: UploadFile) -> Tuple[np.ndarray, bool]:
    """Decode uploaded image file to numpy array"""
    try:
        contents = await uploaded_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        success = img is not None
        return img, success
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None, False

@api_router.post("/merge")
async def merge(
        img_tl: UploadFile = File(...),
        img_tr: UploadFile = File(...),
        img_br: UploadFile = File(...),
        img_bl: UploadFile = File(...),
        path: str = Form(default="visualization/merge.jpg"),
    ):
    
    images = []
    
    try:
        image_files = [img_tl, img_tr, img_br, img_bl]
        
        for i, img_file in enumerate(image_files):
            img, success = await decode_image(img_file)
            if not success:
                return JSONResponse(
                    content={"error": f"Failed to decode image {i+1}."},
                    status_code=400
                )
            
            images.append(img)
         
        # Perform merge and time it
        t1 = time.time()
        merged_img = tcapture_handler.run(*images, path=path)
        elapsed = time.time() - t1
        
        logger.info(f'Inference API time: {elapsed:.3f}s')
        
        return JSONResponse(
            content={"path": path},
            status_code=200
            )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return JSONResponse(
            content={"error": "Internal server error."},
            status_code=500
        )