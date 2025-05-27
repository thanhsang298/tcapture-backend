import os
import time
import json
import numpy as np
import cv2
from typing import Tuple
from PIL import Image, PngImagePlugin
import piexif


class Merger:
    """Efficiently merge four corner images into a single panoramic image with optional metadata saving."""

    def __init__(self, config, save_metadata: bool = True):
        self.config = config
        self.debug = getattr(self.config.Corners, 'DEBUG', False)
        self.vis_path = getattr(self.config.Corners, 'VIS_PATH', '')
        self.img_width = getattr(self.config.API, 'IMG_WIDTH', None)
        self.img_height = getattr(self.config.API, 'IMG_HEIGHT', None)
        self.save_metadata = save_metadata
        # Define crop rectangles (x1, y1, x2, y2) for each corner
        self.crops = {
            'tl': tuple(self.config.Corners.TL),
            'tr': tuple(self.config.Corners.TR),
            'br': tuple(self.config.Corners.BR),
            'bl': tuple(self.config.Corners.BL),
        }

    def crop_images(self, imgs: dict) -> dict:
        """Crop images by key: 'tl','tr','br','bl'. Returns dict of cropped arrays."""
        return {k: arr[y1:y2, x1:x2]
                for (k, arr), (x1, y1, x2, y2) in zip(imgs.items(), self.crops.values())}

    def merge_canvas(self, cropped: dict) -> np.ndarray:
        """Stack cropped corner images into a single canvas array."""
        h, w = next(iter(cropped.values())).shape[:2]
        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = cropped['tl']
        canvas[0:h, w:2*w] = cropped['tr']
        canvas[h:2*h, w:2*w] = cropped['br']
        canvas[h:2*h, 0:w] = cropped['bl']
        return canvas

    def save_with_metadata(self, canvas: np.ndarray, path: str, processing_time: float):
        """Convert numpy (BGR) to PIL (RGB), add metadata, and save in one pass."""
        # Convert BGR to RGB before PIL
        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_canvas)
        metadata = {
            "TotalProcessingTime": f"{processing_time:.4f} seconds",
            "SourceImages": "tl, tr, br, bl",
            "MergedImageType": "Panoramic",
            "Framework": "NumPy-Pillow"
        }

        ext = os.path.splitext(path)[1].lower()
        if ext == '.png':
            png_info = PngImagePlugin.PngInfo()
            for k, v in metadata.items():
                png_info.add_text(k, v)
            img.save(path, pnginfo=png_info)

        elif ext in {'.jpg', '.jpeg'}:
            # Build EXIF dict
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
            # Manually encode UserComment: ASCII prefix + JSON bytes
            comment_bytes = json.dumps(metadata).encode('utf-8')
            ascii_prefix = b'ASCII\x00\x00\x00'
            user_comment = ascii_prefix + comment_bytes
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
            # Optionally add Software tag
            exif_dict["0th"][piexif.ImageIFD.Software] = "MergerScript_v2"
            exif_bytes = piexif.dump(exif_dict)
            img.save(path, exif=exif_bytes, quality=95)

        else:
            img.save(path)

    def run(
        self,
        tl_arr: np.ndarray,
        tr_arr: np.ndarray,
        br_arr: np.ndarray,
        bl_arr: np.ndarray,
        path: str = ""
    ) -> np.ndarray:
        """
        Crop, merge, and optionally save four corner images as a panorama with metadata.

        Args:
            tl_arr, tr_arr, br_arr, bl_arr: NumPy arrays of corner images.
            path: Output file path (requires extension .png/.jpg).

        Returns:
            The merged NumPy array.
        """
        start = time.time()

        # Batch crop and merge
        imgs = {'tl': tl_arr, 'tr': tr_arr, 'br': br_arr, 'bl': bl_arr}
        crops = self.crop_images(imgs)
        canvas = self.merge_canvas(crops)

        # Optional resize
        if self.img_width and self.img_height:
            canvas = cv2.resize(canvas, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)

        # Save with or without metadata
        if path:
            if self.save_metadata:
                self.save_with_metadata(canvas, path, time.time() - start)
            else:
                # Direct save via OpenCV (BGR)
                cv2.imwrite(path, canvas)

        # Debug visualization (convert to RGB for correct colors)
        if self.debug and self.vis_path:
            os.makedirs(self.vis_path, exist_ok=True)
            for key, arr in crops.items():
                # crops are already BGR; convert for PIL
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(os.path.join(self.vis_path, f"cropped_{key}.png"))
            rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_canvas).save(os.path.join(self.vis_path, "merged.png"))

        return canvas
