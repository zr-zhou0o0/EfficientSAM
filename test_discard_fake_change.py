import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import zipfile

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
    
    # 如果有其他模型，如 SqueezeSAM，可以在此添加
    # models['squeeze_sam'] = build_squeeze_sam()
    
    return models

def prepare_model(model, weights_path, device):
    """
    将模型加载到指定设备，并设置为评估模式。
    
    Args:
        model (torch.nn.Module): 要加载的模型。
        weights_path (str): 权重文件的路径。
        device (torch.device): 设备（CPU或GPU）。
    
    Returns:
        torch.nn.Module: 准备好的模型。
    """
    # 加载权重
    # state_dict = torch.load(weights_path, map_location=device)
    # model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_images_from_directory(directory, preprocess, device):
    """
    加载指定目录下的所有图像，并返回一个字典
    键为文件名，值为图像张量
    
    Args:
        directory (str): 图像目录路径。
        preprocess (torchvision.transforms.Compose): 图像预处理变换。
        device (torch.device): 设备（CPU或GPU）。
    
    Returns:
        dict: 包含图像张量的字典。
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
    return images

def compute_embeddings(images, model, device):
    """
    计算所有图像的嵌入，并返回一个字典
    键为文件名，值为嵌入张量
    
    Args:
        images (dict): 包含图像张量的字典。
        model (torch.nn.Module): 用于生成嵌入的模型。
        device (torch.device): 设备（CPU或GPU）。
    
    Returns:
        dict: 包含嵌入张量的字典。
    """
    embeddings = {}
    with torch.no_grad():
        for filename, image_tensor in tqdm(images.items(), desc="计算嵌入"):
            # 添加batch维度
            image_tensor = image_tensor.unsqueeze(0)  # 形状 [1, C, H, W]
            
            # 调用模型前向传播，input_points和input_labels设为None
            # 假设模型的forward方法已被修改以返回嵌入
            image_embedding = model(
                image_tensor,
                None,
                None,
            )  # 预期形状 [1, 256, 64, 64]
            
            # 确保返回的是单个张量
            if isinstance(image_embedding, tuple):
                image_embedding = image_embedding[0]
            
            # 移除batch维度并归一化
            embedding = image_embedding.squeeze(0)  # 形状 [256, 64, 64]
            embedding = F.normalize(embedding.view(256, -1), dim=1)  # 形状 [256, 4096]
            embeddings[filename] = embedding
    return embeddings

def compute_cosine_similarity(embeddings1, embeddings2):
    """
    计算两组嵌入之间的余弦相似度，并返回一个嵌套字典
    NOTE 这里还不是pre和post的两两对应相似度，而是一个相似度邻接矩阵！要改。
    
    Args:
        embeddings1 (dict): 第一组嵌入字典。
        embeddings2 (dict): 第二组嵌入字典。
    
    Returns:
        dict: 嵌套的相似度字典。
    """
    similarities = {}
    for fname1, embed1 in tqdm(embeddings1.items(), desc="计算相似度"):
        similarities[fname1] = {}
        for fname2, embed2 in embeddings2.items():
            # 计算余弦相似度
            # embed1 和 embed2 的形状都是 [256, 4096]
            cos_sim = F.cosine_similarity(embed1, embed2, dim=1).mean().item()
            similarities[fname1][fname2] = cos_sim
    return similarities

def main(dir1, dir2, weights_dir, output_json, model_name='efficientsam_vits'):
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
    models = load_models(weights_dir)
    
    # 检查指定的模型是否存在
    if model_name not in models:
        raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {list(models.keys())}")
    
    # 准备指定的模型
    model = prepare_model(models[model_name], os.path.join(weights_dir, f"{model_name}.pt"), device)
    print(f"已加载模型: {model_name}")
    
    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 根据你的模型输入调整
        transforms.ToTensor(),
        # 添加任何其他需要的预处理步骤
    ])
    
    # 加载目录1的图像
    print("加载目录1的图像...")
    images1 = load_images_from_directory(dir1, preprocess, device)
    print(f"加载了 {len(images1)} 张图像来自 {dir1}")
    
    # 加载目录2的图像
    print("加载目录2的图像...")
    images2 = load_images_from_directory(dir2, preprocess, device)
    print(f"加载了 {len(images2)} 张图像来自 {dir2}")
    
    # 计算目录1的图像嵌入
    print("计算目录1的图像嵌入...")
    embeddings1 = compute_embeddings(images1, model, device)
    
    # 计算目录2的图像嵌入
    print("计算目录2的图像嵌入...")
    embeddings2 = compute_embeddings(images2, model, device)
    
    # 计算余弦相似度
    print("计算余弦相似度...")
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    # 保存结果到JSON文件
    print(f"将结果保存到 {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(similarities, f, indent=4)
    
    print("完成！")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="计算两个目录下图像的余弦相似度并保存为JSON")
    parser.add_argument('--dir1', type=str, default='data/denoise_dif_map/002-1/out-post_masked_image', help='第一个目录路径')
    parser.add_argument('--dir2', type=str, default='data/denoise_dif_map/002-1/out-pre_masked_image', help='第二个目录路径')
    parser.add_argument('--weights_dir', type=str, default='weights', help='模型权重所在的目录')
    parser.add_argument('--output', type=str, default='data/denoise_dif_map/002-1/out_similarities.json', help='输出JSON文件路径')
    parser.add_argument('--model', type=str, default='efficient_sam_vitt', help='使用的模型名称')
    
    args = parser.parse_args()
    
    main(args.dir1, args.dir2, args.weights_dir, args.output, model_name=args.model)
