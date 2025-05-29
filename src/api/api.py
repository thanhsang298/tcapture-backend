import os
import re
import time
import cv2
import numpy as np

from fastapi import FastAPI, APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse, Response
from typing import List, Tuple
from datetime import datetime, date, timedelta

from src.config import TaiyoConfig
from src.handler import TaiyoTcapture
from src.utils.logger import Logger



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
        

@api_router.post("/merge_from_paths")
async def merge_from_paths(
    img_tl_path: str    = Form(...),
    img_tr_path: str    = Form(...),
    img_br_path: str    = Form(...),
    img_bl_path: str    = Form(...),
    merge_path: str     = Form(default="visualization/merge.jpg"),
    start_time: str     = Form(default="00h00m00s"),
    software_time: float = Form(default=10.0),
):
    def parse_time_string(ts: str) -> datetime:
        """
        Parse strings like '10h11m12s' into a datetime for today at 10:11:12.
        If that time is in the future (e.g. after midnight wrap), we subtract one day.
        """
        m = re.match(r'(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+)s)?$', ts)
        if not m:
            raise ValueError(f"Invalid time format: {ts!r}")
        parts = m.groupdict(default="0")
        h, mn, s = int(parts["h"]), int(parts["m"]), int(parts["s"])
        today = date.today()
        dt = datetime(today.year, today.month, today.day, h, mn, s)
        now = datetime.now()
        # if start timestamp ended up in the “future” (past midnight rollover), go back one day
        if dt > now:
            dt -= timedelta(days=1)
        return dt

    images = []
    try:
        #  turn start_time into a datetime
        start_dt = parse_time_string(start_time)

        # load the 4 corner images
        for p in (img_tl_path, img_tr_path, img_br_path, img_bl_path):
            img = cv2.imread(p)
            if img is None:
                return JSONResponse(
                    {"error": f"Failed to read image at {p!r}."},
                    status_code=400
                )
            images.append(img)

        #  run merge and get the time
        t0 = time.time()
        merged_img = tcapture_handler.run(*images, path=merge_path)
        merge_elapsed = time.time() - t0
        software_time += merge_elapsed

        #  compute the TOTAL elapsed from the hardware-start timestamp
        end_dt = datetime.now()
        total_elapsed = (end_dt - start_dt).total_seconds()

        logger.info(f"[API] Merge-only time: {merge_elapsed:.3f}s")
        logger.info(f"[API] End-to-end time since {start_dt.time()}: {total_elapsed:.3f}s")

        # clean up
        for p in (img_tl_path, img_tr_path, img_br_path, img_bl_path):
            try:
                os.remove(p)
                logger.info(f"Removed file: {p}")
            except Exception as e:
                logger.warning(f"Failed to remove {p}: {e}")

        return JSONResponse(
            {
                "merge_path": merge_path,
                "merge_time": round(merge_elapsed, 3),
                "software_time": round(software_time, 3),
                "total_time": round(total_elapsed, 3),
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"[API] Error: {e}", exc_info=True)
        return JSONResponse(
            {"error": "Internal server error."},
            status_code=500
        )
