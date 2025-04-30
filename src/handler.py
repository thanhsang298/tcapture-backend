from src.core import Merger
from src.config import TaiyoConfig
from src.utils.logger import Logger

class TaiyoTcapture:
    """Main handler for Taiyo TCapture operations"""
    
    def __init__(self, config: TaiyoConfig):
        self.config = config
        self.logger = Logger(config)
        self.merger = Merger(config)
        self.logger.info("TaiyoTcapture initialized successfully")

    def run(self, *images, path=""):
        """Process the four corner images and merge them
        
        Args:
            *images: Four numpy arrays representing TL, TR, BR, BL images
            
        Returns:
            numpy.ndarray: Merged and resized image
        """
        if len(images) != 4:
            self.logger.error(f"Expected 4 images, got {len(images)}")
            raise ValueError(f"Expected 4 images, got {len(images)}")
            
        merged_img = self.merger.run(*images, path=path)
        
        return merged_img