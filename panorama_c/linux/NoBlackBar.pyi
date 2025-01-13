from numpy import ndarray

class NoBlackBar:
    image: ndarray
    image_cp: ndarray
    gray_img: ndarray

    def __init__(self, image: ndarray) -> None: ...
    def process(self) -> ndarray: ...
