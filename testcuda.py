import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vits,
    build_efficient_sam_vitt,
)

# Load EfficientSAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_s"
if model_type == "vit_s":
    effsam = build_efficient_sam_vits()
    print("EfficientSAM vit_s loaded")
elif model_type == "vit_t":
    effsam = build_efficient_sam_vitt()
    print("EfficientSAM vit_t loaded")
else:
    raise ValueError("Invalid model type")
effsam.eval()
effsam.to(device)


def effsam_embedding(rgb, upsample=True):
    """
    Get pixel-aligned image embeddings
    @param rgb (HxWx3 np.array or 1x3xHxW tensor): Image
    @param upsample (bool): Whether to upsample the features
    @return features (1xCxHxW tensor): Pixel-aligned image embeddings
    """
    

    if isinstance(rgb, np.ndarray):
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).to(device)[None, ...]
    elif isinstance(rgb, torch.Tensor):
        assert rgb.dim() == 4, "Input tensor should be 1x1xHxW"
        rgb = rgb.to(device)

    # # Define point coordinates for the effsam model.
    # points = torch.tensor([[[[100.0, 150.0], [200.0, 250.0]]]], device=device, dtype=torch.float32)
    # point_labels = torch.tensor([[[1.0, 1.0]]], device=device, dtype=torch.float32)
    # predicted_masks, predicted_iou = effsam(rgb, points.to(device), point_labels.to(device))
    # print("queried!")
    # print("predicted_masks.shape", predicted_masks.shape)
    # print("predicted_iou.shape", predicted_iou.shape)
    # return

    features = effsam.get_image_embeddings(rgb).detach()
    if upsample:
        features = torch.nn.functional.interpolate(
            features, rgb.shape[-2:], mode="bilinear", align_corners=False
        )
    return features

from PIL import Image

image_path = "data/test_imgs/00153.png"
image = np.array(Image.open(image_path))
print("image.shape: ", image.shape)  # image.shape:  (755, 1007, 3)
emb1 = effsam_embedding(image)


# 甚至也不是环境的问题，相同的环境，这个文件夹下就是跑不了？？？？？
# 是不是这个没build 的原因啊 彻底换成3dgscd的环境试一下
# 成功了。说明就是环境的问题。这两段‘efficientsam’的代码不同的。3dgscd用来build efficientsam的代码，和efficientsam的可编辑代码不同。也有可能是因为现场编译的问题。。。？？？？
# 为了换环境换成gscd，把原来的efficient_sam文件夹改名了，换了解释器和运行环境。

# 先把这边的换回来，方便查找区别。 


