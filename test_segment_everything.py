import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io
import cv2
GRID_SIZE = 32


# !pip install git+https://github.com/facebookresearch/segment-anything.git

from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area


def process_small_region(rles):
        '''
        avoid noise and small holes in the mask
        '''
        new_masks = []
        scores = []
        min_area = 100
        nms_thresh = 0.7
        for rle in rles:
            mask = rle_to_mask(rle[0])

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                rles[i_mask] = mask_to_rle_pytorch(mask_torch)
        masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
        return masks


def get_predictions_given_embeddings_and_queries(img, points, point_labels, model):

    print("shape of img", img.shape) # (3, 1200, 1600)
    print("shape of points", points.shape) # (1, 1024, 1, 2), where 1024 is the number of points
    print("shape of point_labels", point_labels.shape) # (1, 1024, 1)

    predicted_masks, predicted_iou = model(img[None, ...], points, point_labels)
    # add a new dimension, like, (1, 3, 1200, 1600)

    print("shape of predicted_iou", predicted_iou.shape) # (1, 1024, 3), where 3 is 3 candidate masks' iou score
    print("shape of predicted_masks", predicted_masks.shape) # (1, 1024, 3, 1200, 1600)

    # this step is to get the max iou in 3 cnadidate for each point! so is still 1024
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True) # sorted ids, is index, like 2,0,1
    # then take the values from the index
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2) # (1, 1024, 3)
    predicted_masks = torch.take_along_dim(predicted_masks, sorted_ids[..., None, None], dim=2) # (1, 1024, 3, 1200, 1600)
    # the none none is to match the dimension of the mask

    # because here we only process one image... so eliminate the first dimension
    predicted_masks = predicted_masks[0] # (1024, 3, 1200, 1600)

    iou = predicted_iou_scores[0, :, 0] # get the max iou score for each point, shape is (1024,)
    index_iou = iou > 0.7 # only keep the points with iou > 0.7
    iou_ = iou[index_iou] # (n,)
    masks = predicted_masks[index_iou] # (n, 3, 1200, 1600)

    score = calculate_stability_score(masks, 0.0, 1.0)
    score = score[:, 0]
    index = score > 0.9
    score_ = score[index]
    masks = masks[index]
    iou_ = iou_[index]

    masks = torch.ge(masks, 0.0) # to make it 0 or 1
    print("shape of masks", masks.shape) # (n, 3, 1200, 1600), n is the number of points, here is 963(of 1024 primary points)
    print("shape of iou_", iou_.shape) # (963,)

    return masks, iou_


def run_everything_ours(img_path, model):
    model = model.cpu()
    # image = cv2.imread(image_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)
    _, original_image_h, original_image_w = img_tensor.shape

    # HERE it is actually evenly spaced points
    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * original_image_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * original_image_h
            xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy))

    points = xy
    num_pts = xy.shape[0]
    point_labels = torch.ones(num_pts, 1)
    with torch.no_grad():
      predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
              img_tensor.cpu(),
              points.reshape(1, num_pts, 1, 2).cpu(),
              point_labels.reshape(1, num_pts, 1).cpu(),
              model.cpu(),
          ) # (963, 3, 1200, 1600)
    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks] 
    # predicted mask: (963, 3, 1200, 1600)
    # m: (3, 1200, 1600)
    # m[0:1]: (1, 1200, 1600) the first mask
    # rle: compressed mask
    predicted_masks = process_small_region(rle) 
    print("predicted_masks: ", predicted_masks) # list of bool matrix, list of masks
    print("len(predicted_masks): ", len(predicted_masks)) # len = 28
    print("predicted_masks[0].shape: ", predicted_masks[0].shape) # (1200, 1600)
    return predicted_masks


def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask[0].shape[0], mask[0].shape[1], 4)) # initialize a white image, RGBA(4 channel). (4, 1200, 1600)
    img[:,:,3] = 0 # set the alpha channel to 0, transparent
    for ann in mask:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]]) # RGB randomly selected between 0 and 1, and alpha channel is 0.5
        img[m] = color_mask # only when the bool matrix is True, the color will be set to img
    ax.imshow(img)


if __name__ == "__main__":

    



    # !git clone https://github.com/yformer/EfficientSAM.git
    # import os
    # os.chdir("EfficientSAM")

    # what can i say man. the reason why i cannot get the mask is vits performs better than vitt.
    
    # from efficient_sam.build_efficient_sam import build_efficient_sam_vits
    from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
    import zipfile

    with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
        zip_ref.extractall("weights")
    efficient_sam_vits_model = build_efficient_sam_vitt()
    efficient_sam_vits_model.eval()

    fig, ax = plt.subplots(1, 2, figsize=(30, 30))
    image_path = "data/test_imgs/00153.png"
    image = np.array(Image.open(image_path))
    ax[0].imshow(image)
    ax[0].title.set_text("Original")
    ax[0].axis('off')

    ax[1].imshow(image)
    mask_efficient_sam_vits = run_everything_ours(image_path, efficient_sam_vits_model) # list of bool matrix
    show_anns_ours(mask_efficient_sam_vits, ax[1])
    ax[1].title.set_text("EfficientSAM")
    ax[1].axis('off')
    # plt.show()
    plt.savefig("data/test_imgs/efficient_sam.png")


    mask = mask_efficient_sam_vits
    fig, ax = plt.subplots(figsize=(15, 15)) 

    mask_only_img = np.zeros((mask[0].shape[0], mask[0].shape[1], 4))  # RGBA

    for ann in mask:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # RGBA
        mask_only_img[m] = color_mask  

    ax.imshow(mask_only_img) 
    ax.axis('off')  
    ax.title.set_text("Mask Only") 
    plt.show() 
    plt.savefig("data/test_imgs/mask_only.png")
