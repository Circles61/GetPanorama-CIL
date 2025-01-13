# cython: language_level=3

# no_black_bar.pyx

import numpy as np
cimport numpy as np
from skimage import io, restoration, color, filters
from skimage.restoration import inpaint
from PIL import ImageFile
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 150000000  # 根据需要调整限制

cdef class NoBlackBar:
    cdef object image
    cdef object image_cp
    cdef object gray_img
    cdef object binary_img
    cdef object blurred_img

    def __init__(self, image):
        self.image = image
        self.image_cp = np.copy(self.image)  # 备份原始图像
        self.gray_img = color.rgb2gray(self.image_cp)  # 转换为灰度图像

        # 对灰度图像进行高斯模糊，消除拼接线
        self.blurred_img = filters.gaussian(self.gray_img, sigma=2)

        # 二值化模糊图像，非零值置为1
        self.binary_img = (self.blurred_img > 0.001).astype(np.uint8) # type: ignore

    cpdef tuple max_inner_rectangle(self):
        cdef int rows, cols
        cdef np.ndarray[np.int32_t, ndim=2] height
        cdef int i, j, current_height, h, w, area, max_area
        cdef list stack
        cdef tuple best_rect

        rows, cols = self.binary_img.shape[0], self.binary_img.shape[1]   # type: ignore # 显式转换为 Python 整数

        # 计算高度矩阵
        height = np.zeros((rows, cols), dtype=np.int32)
        height[0, :] = self.binary_img[0, :] # type: ignore
        for i in range(1, rows):
            height[i, :] = (height[i - 1, :] + 1) * self.binary_img[i, :] # type: ignore

        # 计算最大内接矩形
        max_area = 0
        best_rect = ()

        for i in range(rows):
            stack = []
            for j in range(cols + 1):
                current_height = height[i, j] if j < cols else 0

                while stack and current_height < height[i, stack[-1]]:
                    h = height[i, stack.pop()]
                    w = j if not stack else j - stack[-1] - 1
                    area = h * w
                    if area > max_area:
                        max_area = area
                        best_rect = (i - h + 1, stack[-1] + 1 if stack else 0, i + 1, j)

                stack.append(j)

        return best_rect

    cdef crop_to_rectangle(self):
        rect = self.max_inner_rectangle()
        if rect:
            top, left, bottom, right = rect
            self.image = self.image[top:bottom, left:right] # type: ignore
            self.binary_img = self.binary_img[top:bottom, left:right] # type: ignore

    def process(self):
        self.crop_to_rectangle()

        # 修复剩余小部分黑边
        mask = np.all(self.image == [0, 0, 0], axis=-1)
        inpainted_image = restoration.inpaint.inpaint_biharmonic(self.image, mask, channel_axis=-1)

        # 将图像值归一化到 [0, 1]，然后转换为 uint8 类型
        inpainted_image = (np.clip(inpainted_image, 0, 1) * 255).astype(np.uint8)
        return inpainted_image