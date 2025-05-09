import os
import json
import numpy as np
from PIL import Image
import shutil

def load_difs(difs_npz_path):
    """
    加载difs.npz文件，返回一个字典。
    """
    try:
        difs = np.load(difs_npz_path, allow_pickle=True)
        difs_dict = {key: difs[key] for key in difs.files}
        return difs_dict
    except Exception as e:
        print(f"Error loading difs.npz: {e}")
        exit(1)

def load_mask_image(mask_path):
    """
    加载掩码图像并转换为布尔数组。
    """
    try:
        with Image.open(mask_path) as img:
            img = img.convert('1')  # 转换为二值图像
            mask = np.array(img, dtype=bool)
        return mask
    except FileNotFoundError:
        print(f"Mask file not found: {mask_path}")
        return None
    except Exception as e:
        print(f"Error loading mask image {mask_path}: {e}")
        return None

def compute_iou(mask1, mask2):
    """
    计算两个布尔掩码的交并比（IoU）。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def copy_mask_image(key, seq, pre_masks_dir, post_masks_dir, out_dir, in_dir):
    """
    根据seq的值，将对应的mask图像复制到指定的目录a或b。
    """
    if seq == "pre":
        source_path = os.path.join(pre_masks_dir, f"{key}.png")
        destination_path = os.path.join(out_dir, f"{key}.png")
    elif seq == "post":
        source_path = os.path.join(post_masks_dir, f"{key}.png")
        destination_path = os.path.join(in_dir, f"{key}.png")
    else:
        return  # 如果seq不是pre或post，则不进行复制

    # 检查源文件是否存在
    if not os.path.exists(source_path):
        print(f"Source mask file does not exist: {source_path}. Skipping copy.")
        return

    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(source_path, destination_path)
        print(f"Copied {source_path} to {destination_path}")
    except Exception as e:
        print(f"Error copying {source_path} to {destination_path}: {e}")


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="data/denoise_dif_map/003")
    # parser.add_argument("--difs_npz_path", type=str, help="data/denoise_dif_map/003/difs.npz")
    # parser.add_argument("--pre_masks_dir", type=str, help="data/denoise_dif_map/003/pre_masks")
    # parser.add_argument("--post_masks_dir", type=str, help="data/denoise_dif_map/003/post_masks")
    # parser.add_argument("--output_json_path", type=str, help="data/denoise_dif_map/003/iou_results.json")
    # parser.add_argument("--in_dir", type=str, help="data/denoise_dif_map/003/in_masks")
    # parser.add_argument("--out_dir", type=str, help="data/denoise_dif_map/003/out_masks")
    args = parser.parse_args()

    # difs_npz_path = args.difs_npz_path
    # pre_masks_dir = args.pre_masks_dir
    # post_masks_dir = args.post_masks_dir
    # output_json_path = args.output_json_path
    # in_dir = args.in_dir
    # out_dir = args.out_dir

    root_dir = args.root_dir
    difs_npz_path = os.path.join(root_dir, "difs.npz")
    pre_masks_dir = os.path.join(root_dir, "pre_masks")
    post_masks_dir = os.path.join(root_dir, "post_masks")
    output_json_path = os.path.join(root_dir, "iou_results.json")
    in_dir = os.path.join(root_dir, "in_masks")
    out_dir = os.path.join(root_dir, "out_masks")


    # difs_npz_path = "data/denoise_dif_map/003/difs.npz"
    # pre_masks_dir = "data/denoise_dif_map/003/pre_masks"
    # post_masks_dir = "data/denoise_dif_map/003/post_masks"
    # output_json_path = "data/denoise_dif_map/003/iou_results.json"
    # in_dir = "data/denoise_dif_map/003/in_masks"
    # out_dir = "data/denoise_dif_map/003/out_masks"

    # 阈值设定
    threshold = 0.6 # 0.3 还是会有一些奇怪的东西比如猫头。。哦这个是因为difmap分割造成的。
    # 这个猫头的问题也好解决，先用semantic去掉fakechange之后，把同一张图片合并起来就好啦！

    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 加载difs.npz
    difs_dict = load_difs(difs_npz_path)
    print(f"Loaded {len(difs_dict)} dif maps from {difs_npz_path}")

    results = {}

    for key, dif_map in difs_dict.items():
        print(f"Processing {key}...")

        # 确保dif_map是布尔类型
        if dif_map.dtype != bool:
            dif_map = dif_map.astype(bool)

        # 构建pre和post掩码图像的路径
        pre_mask_path = os.path.join(pre_masks_dir, f"{key}.png")
        post_mask_path = os.path.join(post_masks_dir, f"{key}.png")

        # 加载pre和post掩码图像
        pre_mask = load_mask_image(pre_mask_path)
        post_mask = load_mask_image(post_mask_path)

        # 如果找不到对应的mask图像，则跳过
        if pre_mask is None:
            print(f"Pre mask for {key} not found. Skipping...")
            continue
        if post_mask is None:
            print(f"Post mask for {key} not found. Skipping...")
            continue

        # 确保所有掩码的形状一致
        if dif_map.shape != pre_mask.shape:
            print(f"Shape mismatch for {key}: dif_map {dif_map.shape}, pre_mask {pre_mask.shape}. Skipping...")
            continue
        if dif_map.shape != post_mask.shape:
            print(f"Shape mismatch for {key}: dif_map {dif_map.shape}, post_mask {post_mask.shape}. Skipping...")
            continue

        # 计算IoU
        pre_iou = compute_iou(dif_map, pre_mask)
        post_iou = compute_iou(dif_map, post_mask)

        # 找出最大的IoU及其来源
        if pre_iou >= post_iou:
            max_iou = pre_iou
            seq = "pre"
        else:
            max_iou = post_iou
            seq = "post"

        # 应用阈值
        if max_iou < threshold:
            result = {
                "pre_IoU": pre_iou,
                "post_IoU": post_iou,
                "seq": None,
                "max_IoU": None
            }
            seq = None
            max_iou = None
        else:
            result = {
                "pre_IoU": pre_iou,
                "post_IoU": post_iou,
                "seq": seq,
                "max_IoU": max_iou
            }

        copy_mask_image(key, seq, pre_masks_dir, post_masks_dir, out_dir, in_dir)

        # 保存结果
        results[key] = result

    # 保存结果到JSON文件
    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Results saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

if __name__ == "__main__":
    main()
