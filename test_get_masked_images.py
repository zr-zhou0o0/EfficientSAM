import os
import sys
from PIL import Image
import numpy as np

def extract_imagename(mask_filename):
    """
    从掩码文件名中提取对应的图像名称。
    掩码文件名格式为 {imagename}_{mask_index}.extension

    Args:
        mask_filename (str): 掩码文件名。

    Returns:
        str: 提取的图像名称。
    """
    basename = os.path.splitext(mask_filename)[0]
    imagename = basename.rsplit('_', 1)[0]
    return imagename

def find_image_file(imagename, image_dir, allowed_extensions):
    """
    在图像目录中查找与给定图像名称匹配的图像文件。

    Args:
        imagename (str): 图像名称。
        image_dir (str): 图像目录路径。
        allowed_extensions (list): 允许的图像文件扩展名。

    Returns:
        str or None: 找到的图像文件的完整路径，若未找到则返回 None。
    """
    for ext in allowed_extensions:
        image_filename = f"{imagename}{ext}"
        image_path = os.path.join(image_dir, image_filename)
        if os.path.isfile(image_path):
            return image_path
    return None

def apply_mask_to_image(image, mask):
    """
    根据掩码生成遮罩图像。

    Args:
        image (PIL.Image.Image): 原始图像。
        mask (PIL.Image.Image): 二值掩码图像。

    Returns:
        PIL.Image.Image: 生成的遮罩图像。
    """
    # 确保图像和掩码大小一致
    if image.size != mask.size:
        raise ValueError("图像和掩码的尺寸不一致。")

    # 将图像转换为 NumPy 数组
    image_np = np.array(image)
    
    # 将掩码转换为 NumPy 数组，并确保为二值
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        # 如果掩码是彩色的，取第一个通道
        mask_np = mask_np[:, :, 0]
    mask_binary = (mask_np > 20).astype(np.uint8)  # THRESHOLD = 20

    # 如果图像是灰度图，扩展维度以匹配
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=2)

    # 创建遮罩
    masked_image_np = image_np.copy()
    masked_image_np[mask_binary == 0] = 0  # 掩码为黑色的部分设为黑色

    # 如果原图是灰度图，去除多余的维度
    if masked_image_np.shape[2] == 1:
        masked_image_np = masked_image_np.squeeze(axis=2)

    # 转换回 PIL 图像
    masked_image = Image.fromarray(masked_image_np)
    return masked_image

def process_masks(mask_dir, image_dir, output_dir, allowed_extensions):
    """
    处理掩码文件并生成遮罩图像。

    Args:
        mask_dir (str): 掩码目录路径。
        image_dir (str): 图像目录路径。
        output_dir (str): 输出遮罩图像的目录路径。
        allowed_extensions (list): 允许的图像文件扩展名。
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 获取掩码文件列表
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if os.path.isfile(os.path.join(mask_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    print(f"Processing masks in '{mask_dir}'...")
    for mask_filename in mask_files:
        imagename = extract_imagename(mask_filename)
        image_path = find_image_file(imagename, image_dir, allowed_extensions)

        if image_path is None:
            print(f"Warning: 对于掩码文件 '{mask_filename}'，未找到对应的图像文件 '{imagename}'。跳过。")
            continue

        mask_path = os.path.join(mask_dir, mask_filename)
        output_path = os.path.join(output_dir, mask_filename)

        try:
            # 加载图像和掩码
            image = Image.open(image_path).convert('RGB')  # 保证图像为RGB
            mask = Image.open(mask_path).convert('L')      # 转换为灰度图

            # 生成遮罩图像
            masked_image = apply_mask_to_image(image, mask)

            # 保存遮罩图像
            masked_image.save(output_path)
            print(f"Saved masked image to '{output_path}'")
        except Exception as e:
            print(f"Error processing '{mask_filename}': {e}")

def main():
    # 定义目录路径
    pre_mask_dir = 'data/denoise_dif_map/002-1/in_masks'            # out-pre, in-pre
    post_mask_dir = 'data/denoise_dif_map/002-1/out_masks'          # in-post, out-post
    pre_image_dir = 'data/denoise_dif_map/002-1/pre_normal_images'          # unchange
    post_image_dir = 'data/denoise_dif_map/002-1/post_selected_images'        # unchange
    pre_masked_image_dir = 'data/denoise_dif_map/002-1/in-pre_masked_image'   # out-pre, in-pre   
    post_masked_image_dir = 'data/denoise_dif_map/002-1/out-post_masked_image' # in-post, out-post

    # 定义允许的图像文件扩展名
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tif', '.tiff']

    # 处理预掩码
    process_masks(pre_mask_dir, pre_image_dir, pre_masked_image_dir, allowed_extensions)

    # 处理后掩码
    process_masks(post_mask_dir, post_image_dir, post_masked_image_dir, allowed_extensions)

    print("所有掩码文件已处理完成。")

if __name__ == "__main__":
    main()
