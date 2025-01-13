
"""
请选择全景拼接还是去除黑边
1）全景拼接：将多个图片拼接成一个全景图。
2）去除黑边：将全景图片的黑边去除。

if 1
    请输入所需拼接图片集的文件夹路径
    请输入拼接后的图片保存路径
    开始拼接...
    拼接完成！[或者报错处理]

if 2
    请输入所需去除黑边图片的文件路径
    请输入去除黑边后的图片保存路径/
    开始去除黑边...
    去除黑边完成！[或者报错处理]
"""

from skimage.io import imread, imsave
import os
import time
def main():
    print("""
请选择全景拼接还是去除黑边
1）全景拼接：将多个图片拼接成一个全景图。
2）去除黑边：将全景图片的黑边去除。
    """)
    
    choice = input("请输入选项（1 或 2）：")
    
    try:
        choice = int(choice)
        if choice not in [1, 2]:
            print("[错误] 无效选项。")
            main()
            return
    except ValueError:
        print("[错误] 无效选项。")
        main()
        return
    if int(choice) == 1:
        input_dir = input("请输入所需拼接图片集的文件夹路径：")
        output_path = input("请输入拼接后的图片保存路径：")
        
        if not os.path.isdir(input_dir):
            print("[错误] 输入目录不存在。")
            return
        from panorama.Panorama import Panorama
        panorama = Panorama(input_dir, output_path)
        start_time = time.time()
        panorama.run()
        end_time = time.time()
        print(f"函数运行时间: {end_time - start_time} 秒")
        print("拼接完成！")
        output_path_str = os.path.join(output_path, "panorama_result.jpg")
        print(f"保存在{output_path_str}")
        is_continue()
    elif int(choice) == 2:
        input_path = input("请输入所需去除黑边图片的文件路径：")
        output_path = os.path.join(input("请输入去除黑边后的图片保存路径："), "panorama_result_no_black.png")
        
        if not os.path.isfile(input_path):
            print("[错误] 输入文件不存在。")
            return
        
        image = imread(input_path)
        from panorama.NoBlackBar import NoBlackBar
        no_black_bar = NoBlackBar(image)
        start_time = time.time()
        result_image = no_black_bar.process()
        end_time = time.time()
        print(f"函数运行时间: {end_time - start_time} 秒")
        imsave(output_path, result_image)
        print("去除黑边完成！")
        is_continue()

def is_continue():
    choice = input("是否继续？（y/n）")
    if choice.lower() == "y":
        main()
        return True
    else:
        return False

if __name__ == "__main__":
    main()