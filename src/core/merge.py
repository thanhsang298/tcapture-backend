import cv2
import os
import numpy as np
from src.config import TaiyoConfig
from typing import Tuple, List

class Merger:
    """Class to merge four corner images into a single panoramic image"""
    
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.debug = self.config.Corners.DEBUG
        self.vis_path = self.config.Corners.VIS_PATH
        self.img_width = self.config.API.IMG_WIDTH
        self.img_height = self.config.API.IMG_HEIGHT
        # Store cropping coordinates as y1, x1, y2, x2
        self.tl_crop = self.config.Corners.TL
        self.tr_crop = self.config.Corners.TR 
        self.br_crop = self.config.Corners.BR 
        self.bl_crop = self.config.Corners.BL
    
    def crop_images(self, tl, tr, br, bl) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Crop the four corner images according to configuration"""
        cropped_tl = tl[self.tl_crop[1]:self.tl_crop[3], self.tl_crop[0]:self.tl_crop[2]]
        cropped_tr = tr[self.tr_crop[1]:self.tr_crop[3], self.tr_crop[0]:self.tr_crop[2]]
        cropped_br = br[self.br_crop[1]:self.br_crop[3], self.br_crop[0]:self.br_crop[2]]
        cropped_bl = bl[self.bl_crop[1]:self.bl_crop[3], self.bl_crop[0]:self.bl_crop[2]]
        return cropped_tl, cropped_tr, cropped_br, cropped_bl
        
    
    def run(self, tl, tr, br, bl, path="") -> np.ndarray:
        """
        Process and merge four images into a single panorama
        
        Args:
            tl: Top-left image as numpy array
            tr: Top-right image as numpy array
            br: Bottom-right image as numpy array
            bl: Bottom-left image as numpy array
            
        Returns:
            np.ndarray: Merged and resized image
        """
        cropped_tl, cropped_tr, cropped_br, cropped_bl = self.crop_images(tl, tr, br, bl)
        
        # Create a canvas with double the size
        canvas_height = cropped_tl.shape[0] * 2
        canvas_width = cropped_tl.shape[1] * 2
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Place the cropped images in their respective positions
        h, w = cropped_tl.shape[:2]
        canvas[0:h, 0:w] = cropped_tl  # Top-left
        canvas[0:h, w:2*w] = cropped_tr  # Top-right
        canvas[h:2*h, w:2*w] = cropped_br  # Bottom-right
        canvas[h:2*h, 0:w] = cropped_bl  # Bottom-left
        
        # Save merged image to local path
        cv2.imwrite(path, canvas)

        # Debug: Save intermediate cropped images
        if self.debug:
            os.makedirs(self.vis_path, exist_ok=True)
            cv2.imwrite(f"{self.vis_path}/img_cropped_tl.jpg", cropped_tl)
            cv2.imwrite(f"{self.vis_path}/img_cropped_tr.jpg", cropped_tr)
            cv2.imwrite(f"{self.vis_path}/img_cropped_br.jpg", cropped_br)
            cv2.imwrite(f"{self.vis_path}/img_cropped_bl.jpg", cropped_bl)
            cv2.imwrite(f"{self.vis_path}/merged.jpg", canvas)
        
        return canvas