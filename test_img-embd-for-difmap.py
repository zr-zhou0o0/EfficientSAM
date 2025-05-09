import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import zipfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入模型构建函数
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
# 如果有其他模型，如 SqueezeSAM，可以取消注释并导入
# from squeeze_sam.build_squeeze_sam import build_squeeze_sam

def load_models(weights_dir):
    """
    加载所有需要的模型，并解压权重文件。
    
    Args:
        weights_dir (str): 权重文件所在的目录。
    
    Returns:
        dict: 包含所有模型的字典。
    """
    models = {}
    
    # 构建 EfficientSAM-Ti 模型
    models['efficient_sam_vitt'] = build_efficient_sam_vitt()
    
    # 解压 EfficientSAM-ViTS 权重文件
    vits_zip_path = os.path.join(weights_dir, "efficient_sam_vits.pt.zip")
    vits_extracted_path = os.path.join(weights_dir, "efficient_sam_vits.pt")
    if not os.path.exists(vits_extracted_path):
        with zipfile.ZipFile(vits_zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)
    # 构建 EfficientSAM-ViTS 模型
    models['efficient_sam_vits'] = build_efficient_sam_vits()
    
    return models

def prepare_model(model, weights_path, device):
    model.to(device)
    model.eval()
    return model

def load_images_from_directory(directory, preprocess, device):
    """
    加载指定目录下的所有图像，并返回一个字典
    键为文件名，值为图像张量
    """
    images = {}
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_formats):
            path = os.path.join(directory, filename)
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = preprocess(image).to(device)
                images[filename] = image_tensor
            except Exception as e:
                print(f"无法加载图像 {path}: {e}")
    height, width = image_tensor.shape[-2:]
    return images, height, width

def compute_embeddings(images, model, device):
    embeddings = {}
    with torch.no_grad():
        for filename, image_tensor in tqdm(images.items(), desc="cumpute embeddings"):
            # 添加batch维度
            image_tensor = image_tensor.unsqueeze(0)  # 形状 [1, C, H, W]
            
            # 调用模型前向传播，input_points和input_labels设为None
            # 假设模型的forward方法已被修改以返回嵌入
            image_embedding = model(
                image_tensor,
                None,
                None,
                img_embedding_only=True
            )  # 预期形状 [1, 256, 64, 64]
            
            # 确保返回的是单个张量
            if isinstance(image_embedding, tuple):
                image_embedding = image_embedding[0]
            
            # 移除batch维度并归一化
            embedding = image_embedding.squeeze(0)  # 形状 [256, 64, 64]
            # embedding = F.normalize(embedding.view(256, -1), dim=1)  # 形状 [256, 4096]
            embeddings[filename] = embedding # 形状 [256, 64, 64]
    return embeddings

def compute_cosine_similarity(embeddings1, embeddings2):
   
    similarities = {}
    for fname1, embed1 in tqdm(embeddings1.items(), desc="计算相似度"):
        for fname2, embed2 in embeddings2.items():
            # 计算余弦相似度
            # embed1 和 embed2 的形状都是 [256, 4096]
            if fname2 == fname1:
                cos_sim = F.cosine_similarity(embed1, embed2, dim=1).mean().item()
                similarities[fname1] = cos_sim
                break
    return similarities


def compute_cosine_similarity_pixelwise(embeddings1, embeddings2):
    similarities = {}
    
    for fname1, embed1 in tqdm(embeddings1.items(), desc="cos sim"):
        for fname2, embed2 in embeddings2.items():
            # 只计算同名文件之间的相似度
            if fname2 == fname1:
                # 假设 embed1 和 embed2 都是 256 x 64 x 64 的形状
                cos_sim_map = np.zeros(embed1.shape[1:])  # 初始化一个 64 x 64 的矩阵来存储相似度结果
                
                for i in range(embed1.shape[1]):  # 遍历 64x64 特征图的每个像素
                    for j in range(embed1.shape[2]):
                        # 提取每个像素对应的向量，计算余弦相似度
                        pixel_embed1 = embed1[:, i, j].reshape(1, -1)  # (1, 256)
                        pixel_embed2 = embed2[:, i, j].reshape(1, -1)  # (1, 256)

                        if isinstance(pixel_embed1, torch.Tensor):
                            pixel_embed1 = pixel_embed1.cpu().numpy()
                        if isinstance(pixel_embed2, torch.Tensor):
                            pixel_embed2 = pixel_embed2.cpu().numpy()

                        cos_sim = cosine_similarity(pixel_embed1, pixel_embed2)[0][0]
                        cos_sim_map[i, j] = cos_sim

                # cos_sim_map = cos_sim_map.astype(np.float64)
                # cos_sim_map = torch.tensor(cos_sim_map, dtype=torch.float64)
                # cos_sim_map = cos_sim_map.tolist()
                
                similarities[fname1] = cos_sim_map

    return similarities


def plot_cosine_similarity_heatmaps(similarities, save_path):
    os.makedirs(save_path, exist_ok=True)
    for fname, cos_sim_map in similarities.items():
        # 创建热图
        plt.figure(figsize=(8, 6))
        plt.imshow(cos_sim_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Cosine Similarity Heatmap: {fname}')
        fname = fname.split('.')[0]
        # 保存热图
        plt.savefig(f'{save_path}/{fname}_cos_sim_heatmap.png')
        plt.close()
        print(f"保存热图: {save_path}/{fname}_cos_sim_heatmap.png")
    
    print("热图保存完毕")

def resize_similarities(similarities, target_height, target_width):
    resized_similarities = {}
    
    for fname, cos_sim_map in similarities.items():
        # 将 NumPy 数组转换为 PyTorch 张量
        cos_sim_map_tensor = torch.tensor(cos_sim_map, dtype=torch.float32)
        
        # 使用 F.interpolate 进行上采样（插值）
        resized_map = F.interpolate(cos_sim_map_tensor.unsqueeze(0).unsqueeze(0),  # 增加batch和channel维度
                                    size=(target_height, target_width),  # 目标大小
                                    mode='bilinear',  # 插值模式
                                    align_corners=False)  # 对齐角点（防止插值时有偏差）
        
        # 去掉额外的维度，恢复为 [H, W] 的尺寸
        resized_map = resized_map.squeeze(0).squeeze(0).cpu().numpy()  # 转回 NumPy 数组
        
        # 存储调整后的相似度图
        resized_similarities[fname] = resized_map
    
    return resized_similarities


def get_difference_map(similarities, save_path, threshold):
    """
    生成基于余弦相似度的差异图。

    Args:
        similarities (dict): 余弦相似度字典，键为文件名，值为相似度矩阵（二维数组）。
        save_path (str): 存储差异图的目录。
        threshold (float): 判断为差异的相似度阈值，低于该值认为是差异。
    """
    os.makedirs(save_path, exist_ok=True)
    
    for fname, cos_sim_map in similarities.items():
        # 创建一个二值化的差异图
        diff_map = cos_sim_map < threshold  # 如果相似度低于阈值，则为 True（表示差异）
        
        # 将差异图从布尔型转换为 0 或 1 数值型
        diff_map = diff_map.astype(np.uint8)  # 转换为 0 和 1 的二值图
        
        # 保存差异图为图像
        diff_map_image = Image.fromarray(diff_map * 255)  # 将 0 和 1 映射为 0 和 255，以便保存为图像
        diff_map_image = diff_map_image.convert('L')  # 转为灰度图
        
        # 保存图像
        diff_map_image.save(f"{save_path}/{fname}")
        print(f"保存差异图: {save_path}/{fname}")



def main(args):
    """
    主函数，执行加载模型、计算嵌入、计算相似度并保存结果。
    
    Args:
        dir1 (str): 第一个图像目录路径。
        dir2 (str): 第二个图像目录路径。
        weights_dir (str): 权重文件所在的目录。
        output_json (str): 输出JSON文件路径。
        model_name (str): 使用的模型名称。
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载所有模型
    print("加载模型...")
    models = load_models(args.weights_dir)
    
    # 检查指定的模型是否存在
    if args.model not in models:
        raise ValueError(f"模型 '{args.model}' 不存在。可用模型: {list(models.keys())}")
    
    # 准备指定的模型
    model = prepare_model(models[args.model], os.path.join(args.weights_dir, f"{args.model}.pt"), device)
    print(f"已加载模型: {args.model}")
    
    # 定义图像预处理
    preprocess = transforms.Compose([
        # transforms.Resize((256, 256)),  
        transforms.ToTensor(),
    ])
    
    # 加载目录1的图像
    print("加载目录1的图像...")
    images1, height, width = load_images_from_directory(args.dir1, preprocess, device)
    print(f"加载了 {len(images1)} 张图像来自 {args.dir1}")
    print(f"图像尺寸: {height} x {width}")
    
    # 加载目录2的图像
    print("加载目录2的图像...")
    images2, _, _ = load_images_from_directory(args.dir2, preprocess, device)
    print(f"加载了 {len(images2)} 张图像来自 {args.dir2}")
    
    # 计算目录1的图像嵌入
    print("计算目录1的图像嵌入...")
    embeddings1 = compute_embeddings(images1, model, device)
    
    # 计算目录2的图像嵌入
    print("计算目录2的图像嵌入...")
    embeddings2 = compute_embeddings(images2, model, device)
    
    # 计算余弦相似度
    print("计算余弦相似度...")
    similarities = compute_cosine_similarity_pixelwise(embeddings1, embeddings2)

    similarities = resize_similarities(similarities, height, width)

    print("similarities:", similarities)

    plot_cosine_similarity_heatmaps(similarities, args.heatmap)

    get_difference_map(similarities, args.difmap, args.threshold)
    
    # 保存结果到JSON文件
    # print(f"将结果保存到 {output_json}...")
    # with open(output_json, 'w') as f:
    #     json.dump(similarities, f, indent=4)
    
    print("Done")

if __name__ == "__main__":
    import argparse
    '''
    test:
    python test_img-embd-for-difmap.py --dir1 data/dif_map/003/testpre --dir2 data/dif_map/003/testpost --heatmap data/dif_map/003/heatmap --output data/dif_map/003/cos.json --difmap data/dif_map/003/difmap --threshold 0.27

    003
    python test_img-embd-for-difmap.py --dir1 data/denoise_dif_map/003/pre_normal_images --dir2 data/denoise_dif_map/003/post_selected_images --heatmap data/dif_map/003/heatmap --difmap data/dif_map/003/difmap --threshold 0.27

    
    python test_img-embd-for-difmap.py --dir1 data/denoise_dif_map/007_card/renders --dir2 data/denoise_dif_map/007_card/train-pre --heatmap data/denoise_dif_map/007/heatmap --difmap data/denoise_dif_map/007/difmap --threshold 0.27
    
    '''
    
    parser = argparse.ArgumentParser(description="计算两个目录下图像的余弦相似度并保存为JSON")
    parser.add_argument('--dir1', type=str, help='第一个目录路径')
    parser.add_argument('--dir2', type=str, help='第二个目录路径')
    parser.add_argument('--heatmap', type=str, help='保存热图的路径')
    # parser.add_argument('--output', type=str, help='输出JSON文件路径')
    parser.add_argument('--difmap', type=str, help='输出difference map文件路径')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--weights_dir', type=str, default='weights', help='模型权重所在的目录')
    parser.add_argument('--model', type=str, default='efficient_sam_vitt', help='使用的模型名称')
    
    args = parser.parse_args()
    
    main(args)
