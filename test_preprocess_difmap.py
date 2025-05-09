import os
import cv2
import numpy as np
from collections import OrderedDict

def process_images(input_dir, output_dir, mask_output_path, area_threshold=500):
    """
    处理输入目录中的灰度图像，去除小区域噪声，并保存主要mask到输出目录。

    Args:
        input_dir (str): 输入图像目录，包含灰度图像。
        output_dir (str): 输出图像目录，用于保存处理后的mask图像。
        mask_output_path (str): 保存切割好的mask矩阵的npy或npz文件路径。
        area_threshold (int): 面积阈值，小于此值的区域将被视为噪声并去除。
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录中的所有文件，并排序
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])

    all_masks = OrderedDict()

    for image_file in image_files:
        # 读取图像
        input_path = os.path.join(input_dir, image_file)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"无法读取图像：{input_path}")
            continue

        # 二值化图像（假设白色部分是255，黑色部分是0）
        _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个空白图像，用于绘制主要mask
        cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)

        # 保存单个图像的所有mask矩阵
        individual_masks = []

        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)

            # 保留面积大于阈值的轮廓
            if area > area_threshold:
                # 创建独立的mask
                single_mask = np.zeros_like(binary_mask, dtype=np.uint8)
                cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)
                individual_masks.append(single_mask.astype(bool))

                # 在清理后的mask图像中标记不同的灰度值
                color = (i*41) % 200 + 54 # XXX haha very hash!
                # cv2.drawContours(cleaned_mask, [contour], -1, i + 1, thickness=cv2.FILLED)
                cv2.drawContours(cleaned_mask, [contour], -1, color, thickness=cv2.FILLED)

        # 保存处理后的图像
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, cleaned_mask)

        # 为每个分割mask分配唯一索引
        for idx, mask in enumerate(individual_masks):
            mask_name = f"{os.path.splitext(image_file)[0]}_dif{idx}"
            all_masks[mask_name] = mask

        print(f"已处理并保存图像：{output_path}")

    # 保存所有mask矩阵到npy/npz文件
    np.savez(mask_output_path, **all_masks)
    print(f"所有mask矩阵已保存到：{mask_output_path}")



import argparse

parser = argparse.ArgumentParser(description="处理difmap图像")
parser.add_argument("--input_directory", type=str, help="输入difmap图像目录")
parser.add_argument("--output_directory", type=str, help="输出difmap图像目录")
parser.add_argument("--mask_output_file", type=str, help="保存difmap mask的npy或npz文件路径")

# input_directory = "data/dif_map/003/difmap"  
# output_directory = "data/denoise_dif_map/003/dif_images"  
# mask_output_file = "data/denoise_dif_map/003/difs.npz"  

args = parser.parse_args()
input_directory = args.input_directory
output_directory = args.output_directory
mask_output_file = args.mask_output_file

process_images(input_directory, output_directory, mask_output_file, area_threshold=500)
