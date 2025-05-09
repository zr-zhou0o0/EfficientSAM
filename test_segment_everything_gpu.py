'''
fuckthat. it cannot run on gpu!!!
just vitt with 16x16 grid size can run on gpu.
'''


import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io
import cv2
# GRID_SIZE =128 # 256 is ok, 400 not.
GRID_SIZE = 32

from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area

def process_small_region(rles, device):
    """
    Avoid noise and small holes in the mask.
    """
    new_masks = []
    scores = []
    min_area = 100
    nms_thresh = 0.7

    for rle in rles:
        mask = torch.from_numpy(rle_to_mask(rle[0]).astype(np.uint8)).to(device)

        # Convert mask to NumPy for compatibility with remove_small_regions
        mask_np = mask.cpu().numpy()

        mask_np, changed = remove_small_regions(mask_np, min_area, mode="holes")
        unchanged = not changed
        mask_np, changed = remove_small_regions(mask_np, min_area, mode="islands")
        unchanged = unchanged and not changed

        # Convert back to PyTorch Tensor
        mask = torch.from_numpy(mask_np).to(device)

        new_masks.append(mask.unsqueeze(0))
        # Give score=0 to changed masks and score=1 to unchanged masks
        scores.append(float(unchanged))

    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores, device=device),
        torch.zeros_like(boxes[:, 0], device=device),  # categories
        iou_threshold=nms_thresh,
    )

    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_torch = masks[i_mask].unsqueeze(0)
            rles[i_mask] = mask_to_rle_pytorch(mask_torch)
    masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
    return masks

def get_predictions_given_embeddings_and_queries(img, points, point_labels, model):
    """Run prediction on GPU."""
    print("shape of img", img.shape)  # (3, 1200, 1600)
    print("shape of points", points.shape)  # (1, 1024, 1, 2)
    print("shape of point_labels", point_labels.shape)  # (1, 1024, 1)

    predicted_masks, predicted_iou = model(img[None, ...], points, point_labels)

    print("shape of predicted_iou", predicted_iou.shape)  # (1, 1024, 3)
    print("shape of predicted_masks", predicted_masks.shape)  # (1, 1024, 3, 1200, 1600)

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_masks = torch.take_along_dim(predicted_masks, sorted_ids[..., None, None], dim=2)

    predicted_masks = predicted_masks[0]

    iou = predicted_iou_scores[0, :, 0]
    index_iou = iou > 0.7
    iou_ = iou[index_iou]
    masks = predicted_masks[index_iou]

    score = calculate_stability_score(masks, 0.0, 1.0)
    score = score[:, 0]
    index = score > 0.9
    score_ = score[index]
    masks = masks[index]
    iou_ = iou_[index]

    masks = torch.ge(masks, 0.0)
    print("shape of masks", masks.shape)  # (n, 3, 1200, 1600)
    print("shape of iou_", iou_.shape)  # (963,)

    return masks, iou_

def run_everything_ours(img_path, model, device):
    model = model.to(device)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image).to(device)
    _, original_image_h, original_image_w = img_tensor.shape

    # 打开灰度图像
    image = Image.open("data/test_imgs/00153_mask.png").convert("L")
    # 把图像转换为bool，白色区域为True，黑色为False
    mask_bool = np.array(image) > 128


    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * original_image_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * original_image_h
            # 只添加在mask_bool区域内的点
            if mask_bool[int(curr_y), int(curr_x)]:
                xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy)).to(device)

    points = xy
    num_pts = xy.shape[0]
    point_labels = torch.ones(num_pts, 1, device=device)

    with torch.no_grad():
        predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
            img_tensor,
            points.reshape(1, num_pts, 1, 2),
            point_labels.reshape(1, num_pts, 1),
            model
        )

    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
    predicted_masks = process_small_region(rle, device)
    # print("predicted_masks: ", predicted_masks)
    print("len(predicted_masks): ", len(predicted_masks))
    print("predicted_masks[0].shape: ", predicted_masks[0].shape)
    return predicted_masks

def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask[0].shape[0], mask[0].shape[1], 4))
    img[:, :, 3] = 0
    for ann in mask:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)

if __name__ == "__main__":
    # from efficient_sam.build_efficient_sam import build_efficient_sam_vits
    from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
    import zipfile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    #     zip_ref.extractall("weights")
    # efficient_sam_vits_model = build_efficient_sam_vits().to(device)
    efficient_sam_vits_model = build_efficient_sam_vitt().to(device)

    efficient_sam_vits_model.eval()

    fig, ax = plt.subplots(1, 2, figsize=(30, 30))
    image_path = "data/test_imgs/00153.png"
    image = np.array(Image.open(image_path))
    ax[0].imshow(image)
    ax[0].title.set_text("Original")
    ax[0].axis('off')

    ax[1].imshow(image)
    mask_efficient_sam_vits = run_everything_ours(image_path, efficient_sam_vits_model, device)
    show_anns_ours(mask_efficient_sam_vits, ax[1])
    ax[1].title.set_text("EfficientSAM")
    ax[1].axis('off')
    plt.savefig("data/test_imgs/efficient_sam.png")

    mask = mask_efficient_sam_vits
    fig, ax = plt.subplots(figsize=(15, 15))
    mask_only_img = np.zeros((mask[0].shape[0], mask[0].shape[1], 4))

    for ann in mask:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        mask_only_img[m] = color_mask

    ax.imshow(mask_only_img)
    ax.axis('off')
    ax.title.set_text("Mask Only")
    plt.show()
    plt.savefig("data/test_imgs/mask_only.png")
