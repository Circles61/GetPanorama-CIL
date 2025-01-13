# cython: language_level=3
import numpy as np
import cython
cimport numpy as np
from skimage import io, color, feature
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp
from skimage.util import img_as_ubyte
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


cdef class Panorama:
    """自动全景拼接"""
    cdef:
        object input_dir
        object output_dir
        list images
        list grayscale_images
        list keypoints
        list descriptors
        list transforms
        list extensions

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.extensions = ['jpg', 'png', 'jpeg']
        self.images = []
        self.grayscale_images = []
        self.keypoints = []
        self.descriptors = []
        self.transforms = [ProjectiveTransform()]

        if not self.input_dir.is_dir():
            raise FileNotFoundError("[错误] 输入目录不存在。")
        if not self.output_dir.is_dir():
            raise FileNotFoundError("[错误] 输出目录不存在。")

    cdef load_images(self):
        """加载图像并转换为灰度"""
        cdef str ext
        for ext in self.extensions:
            self.images.extend(self.input_dir.glob(f"*.{ext}")) # type: ignore
        if len(self.images) < 2:
            raise ValueError("[错误] 至少需要两张图像进行拼接。")
        self.images = [io.imread(img) for img in self.images]
        self.grayscale_images = [
            color.rgb2gray(img) if img.ndim == 3 else img for img in self.images
        ]

    @cython.wraparound(False) # type: ignore
    @cython.boundscheck(False) # type: ignore
    cdef extract_features(self):
        """提取特征点和描述符"""
        cdef np.ndarray img
        for img in self.grayscale_images:
            sift = feature.SIFT()
            sift.detect_and_extract(img)
            self.keypoints.append(sift.keypoints)
            self.descriptors.append(sift.descriptors)



    cdef match_features(self, desc1, desc2):
        """匹配两个图像的特征点"""
        matches = match_descriptors(desc1, desc2, cross_check=True)
        return matches

    cdef compute_transforms(self):
        """计算投影变换矩阵"""
        cdef np.ndarray src, dst
        cdef int i
        for i in range(1, len(self.grayscale_images)):
            matches = self.match_features(
                self.descriptors[i - 1], self.descriptors[i]
            )
            src = self.keypoints[i][matches[:, 1]][:, ::-1]
            dst = self.keypoints[i - 1][matches[:, 0]][:, ::-1]
            model, inliers = ransac(
                (src, dst),
                ProjectiveTransform,
                min_samples=4,
                residual_threshold=2,
                max_trials=500,
            )
            self.transforms.append(
                ProjectiveTransform(model.params @ self.transforms[-1].params) # type: ignore
            )

    cdef stitch_images(self):
        """拼接图像"""
        cdef list corners
        cdef np.ndarray[np.float32_t, ndim=3] panorama
        cdef np.ndarray[np.float32_t, ndim=2] weights
        cdef object offset
        cdef int r, c

        # 计算图像变换后的角点范围
        corners = []
        cdef np.ndarray img
        for _, (img, transform) in enumerate(zip(self.images, self.transforms)):
            r, c = img.shape[:2] # type: ignore
            corners.append(
                transform([[0, 0], [0, r], [c, 0], [c, r]])
            )
        cdef np.ndarray corners_npar = np.vstack(corners)
        min_coords = corners_npar.min(axis=0)
        max_coords = corners_npar.max(axis=0)
        output_shape = (np.ceil(max_coords - min_coords)[::-1]).astype(int) # type: ignore

        # 创建全景图和权重矩阵
        panorama = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.float32)
        weights = np.zeros((output_shape[0], output_shape[1]), dtype=np.float32)

        offset = SimilarityTransform(translation=-min_coords) # type: ignore
        cdef int i
        for i, (img, transform) in enumerate(zip(self.images, self.transforms)):
            warp_img = warp(
                img, (transform + offset).inverse, output_shape=output_shape[:2]
            )
            mask = (warp_img > 0).astype(np.float32) # type: ignore
            mask = mask.mean(axis=-1)   # type: ignore # 将三通道的 mask 转换为单通道
            weights += mask
            k = warp_img * mask[..., np.newaxis] # type: ignore
            if not np.all(panorama == 0):
                """处理掉鬼影与割裂的效果"""
                overlap_mask = panorama[:, :, 0] != 0 # type: ignore
                for c in range(panorama.shape[2]):   # type: ignore # 遍历每个颜色通道
                    panorama[overlap_mask, c] = (  # type: ignore
                        panorama[overlap_mask, c] * (1 - mask[overlap_mask]) +  # type: ignore
                        k[overlap_mask, c] * mask[overlap_mask]  # type: ignore
                    )  # type: ignore
                panorama[panorama == 0] += k[panorama == 0]  # type: ignore
            else:
                panorama += k
        return img_as_ubyte(panorama)

    cdef save_panorama(self, np.ndarray[np.uint8_t, ndim=3] panorama):
        """保存拼接结果"""
        output_path = self.output_dir / "panorama_result.jpg"  # type: ignore
        io.imsave(output_path, panorama)
        print(f"[INFO] 全景图保存到 {output_path}")

    def run(self):
        """运行全景拼接流程"""
        print("[INFO] 加载图像...")
        self.load_images()
        print("[INFO] 提取特征点...")
        self.extract_features()
        print("[INFO] 计算变换矩阵...")
        self.compute_transforms()
        print("[INFO] 拼接图像...")
        panorama = self.stitch_images()
        print("[INFO] 保存拼接结果...")
        self.save_panorama(panorama)
        print("[INFO] 全景拼接完成！")
