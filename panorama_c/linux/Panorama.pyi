# Panorama.pyi

from pathlib import Path
from typing import List
from numpy import ndarray
from skimage.transform import ProjectiveTransform

class Panorama:
    """自动全景拼接"""

    input_dir: Path
    output_dir: Path
    extensions: List[str]
    images: List[ndarray]
    grayscale_images: List[ndarray]
    keypoints: List[ndarray]
    descriptors: List[ndarray]
    transforms: List[ProjectiveTransform]

    def __init__(self, input_dir: str, output_dir: str) -> None:
        ...

    def run(self) -> None:
        """运行全景拼接流程"""
        ...