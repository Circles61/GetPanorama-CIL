from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# 定义扩展模块
extensions = [
    Extension(
        "Panorama",  # 模块名称
        ["Panorama.pyx"],  # Cython代码文件
        include_dirs=[np.get_include()],  # 添加NumPy头文件路径
    )
]

# 设置编译选项
setup(
    name="Panorama",  # 包名
    version="0.1",  # 版本号
    ext_modules=cythonize(extensions),  # 使用Cython编译扩展模块
    install_requires=[
        "numpy",
        "scikit-image",
        "matplotlib",
    ],  # 依赖项
)

# Panorama
# python setup.py build_ext --inplace