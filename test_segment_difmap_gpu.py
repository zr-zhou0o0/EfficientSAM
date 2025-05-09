import matplotlib.pyplot as plt 
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io
import cv2
GRID_SIZE = 64 # 32
import argparse
from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt
import zipfile
import os

# !pip install git+https://github.com/facebookresearch/segment-anything.git

from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area


def process_small_region(rles, device):
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

        new_masks.append(torch.as_tensor(mask).unsqueeze(0).to(device))
        # Give score=0 to changed masks and score=1 to unchanged masks
        # so NMS will prefer ones that didn't need postprocessing
        scores.append(float(unchanged))

    if len(new_masks) == 0:
        return []

    # Recalculate boxes and remove any new duplicates # Here! 963 to 28.
    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores).to(device),
        torch.zeros_like(boxes[:, 0]).to(device),  # categories
        iou_threshold=nms_thresh,
    )

    # Only recalculate RLEs for masks that have changed
    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_torch = masks[i_mask].unsqueeze(0)
            rles[i_mask] = mask_to_rle_pytorch(mask_torch)
    masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
    return masks


def get_predictions_given_embeddings_and_queries(img, points, point_labels, model, device):

    print("shape of img", img.shape) # (3, 1200, 1600)
    print("shape of points", points.shape) # (1, 1024, 1, 2), where 1024 is the number of points
    print("shape of point_labels", point_labels.shape) # (1, 1024, 1)

    predicted_masks, predicted_iou = model(img[None, ...].to(device), points.to(device), point_labels.to(device))
    # add a new dimension, like, (1, 3, 1200, 1600)

    print("shape of predicted_iou", predicted_iou.shape) # (1, 1024, 3), where 3 is 3 candidate masks' iou score
    print("shape of predicted_masks", predicted_masks.shape) # (1, 1024, 3, 1200, 1600)

    # this step is to get the max iou in 3 candidate for each point! so is still 1024
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True) # sorted ids, is index, like 2,0,1
    # then take the values from the index
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2) # (1, 1024, 3)
    predicted_masks = torch.take_along_dim(predicted_masks, sorted_ids[..., None, None].expand_as(predicted_masks), dim=2) # (1, 1024, 3, 1200, 1600)
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



def get_predictions_with_batching(img, points, point_labels, model, device, batch_size=100):
    """
    如果查询点超过batch_size，则将其分割为多个批次，分别查询模型并合并结果。

    Args:
        img (Tensor): 输入图像张量。
        points (Tensor): 查询点张量，形状为 (1, N, 1, 2)。
        point_labels (Tensor): 查询点标签张量，形状为 (1, N, 1)。
        model (nn.Module): 用于预测的模型。
        device (torch.device): 计算设备。
        batch_size (int): 每个批次的最大查询点数。

    Returns:
        Tuple[Tensor, Tensor]: 合并后的mask张量和iou分数张量。
    """
    num_pts = points.shape[1]
    if num_pts <= batch_size:
        return get_predictions_given_embeddings_and_queries(img, points, point_labels, model, device)
    
    # 初始化列表以存储每个批次的结果
    masks_list = []
    iou_list = []
    
    for start in range(0, num_pts, batch_size):
        end = start + batch_size
        batch_points = points[:, start:end, :, :]
        batch_labels = point_labels[:, start:end, :]
        
        masks, iou = get_predictions_given_embeddings_and_queries(img, batch_points, batch_labels, model, device)
        
        # 检查当前批次是否有有效的mask
        if masks.numel() > 0:
            masks_list.append(masks)
            iou_list.append(iou)
    
    if len(masks_list) == 0:
        # 返回空张量，确保与原函数返回类型一致
        return torch.empty(0, device=device), torch.empty(0, device=device)
    
    # 将所有批次的mask和iou拼接起来
    combined_masks = torch.cat(masks_list, dim=0)
    combined_iou = torch.cat(iou_list, dim=0)
    
    return combined_masks, combined_iou





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

    '''
    0. prepare: dif_images(denoised & segmented dif map), difs.npz, pre_images, post_images (pre_normal_images & post_selected_images). all name matched.

    1. combine many image to a batch (give up)
    2. for each dif in difs.npz(~300), get a enabled grid whose point in dif map has 'True' value. save the query point list.
    3. for each group of query point, run everything ours and get the mask list.
    4. compute mask list and dif's IoU and get the best mask for each dif.
    5. save the best 300 pre_masks and post_masks.

    4. *get the semantic feature of each mask,*
    5. *use cosine similarity, cluster to count the number for each class, get the valid class and assign a unique key to each class. for every image, there may be mask or not, every mask belongs to a class.*
    6. *for 'out' object,*
    '''

    # !git clone https://github.com/yformer/EfficientSAM.git
    # import os
    # os.chdir("EfficientSAM")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--difs_npz_path", type=str, help="data/denoise_dif_map/003/difs.npz")
    parser.add_argument("--pre_images_path", type=str, help="data/denoise_dif_map/003/pre_normal_images")
    parser.add_argument("--post_images_path", type=str, help="data/denoise_dif_map/003/post_selected_images")
    parser.add_argument("--pre_mask_path", type=str, default="data/denoise_dif_map/003/pre_masks")
    parser.add_argument("--post_mask_path", type=str, default="data/denoise_dif_map/003/post_masks")
    parser.add_argument("--original_image_w", type=int, default=1600)
    parser.add_argument("--original_image_h", type=int, default=1200)
    parser.add_argument("--start_i", type=int, default=0)
    parser.add_argument("--seq", type=str, default="pre,post")
    args = parser.parse_args()


    # 定义设备为 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # batchsize = 4
    # difs_npz_path = "data/denoise_dif_map/003/difs.npz"
    # pre_images_path = "data/denoise_dif_map/003/pre_normal_images"
    # post_images_path = "data/denoise_dif_map/003/post_selected_images"
    # pre_mask_path = "data/denoise_dif_map/003/pre_masks"
    # post_mask_path = "data/denoise_dif_map/003/post_masks"
    # original_image_w = 1600
    # original_image_h = 1200
    # start_i = 0 # 0, for 手动断点续传。。。# XXX
    # failure i: 40
    # seq = ["pre","post"]
    # seq = ["post"]

    batchsize = args.batchsize
    difs_npz_path = args.difs_npz_path
    pre_images_path = args.pre_images_path
    post_images_path = args.post_images_path
    pre_mask_path = args.pre_mask_path
    post_mask_path = args.post_mask_path
    original_image_w = args.original_image_w
    original_image_h = args.original_image_h
    start_i = args.start_i
    if args.seq == "pre":
        seq = ["pre"]
    elif args.seq == "post":
        seq = ["post"]
    elif args.seq == "pre,post" or args.seq == "post,pre":
        seq = ["pre","post"]
    else:
        raise ValueError("seq must be 'pre' or 'post' or 'pre,post' or 'post,pre'")



    
    # with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    #     zip_ref.extractall("weights")
    # efficient_sam_vits_model = build_efficient_sam_vits()
    # efficient_sam_vits_model.eval()
    # model = efficient_sam_vits_model.to(device)  # 移动模型到 GPU

    efficient_sam_vitt_model = build_efficient_sam_vitt()
    efficient_sam_vitt_model.eval()
    model = efficient_sam_vitt_model.to(device)


    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * original_image_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * original_image_h
            xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy)).to(device)  # 将 xy 移动到 GPU
    print("xy.shape: ", xy.shape) # (1024, 2) if GRID_SIZE = 32
    len_xy = xy.shape[0]

    difs = np.load(difs_npz_path)
    difkeys = list(difs.keys())
    print("difs keys: ", difkeys)
    num_difs = len(difkeys)

    # pre
    pre_images_names = sorted([name.split(".")[0] for name in os.listdir(pre_images_path)])
    print("pre_images_names: ", pre_images_names)

    # post
    post_images_names = sorted([name.split(".")[0] for name in os.listdir(post_images_path)])
    print("post_images_names: ", post_images_names)

    # iter_num = num_difs // batchsize if num_difs % batchsize == 0 else num_difs // batchsize + 1 # e.g. num_difs = 17, batchsize = 4, iter_num = 5
    pre_skip_i = []
    post_skip_i = []

    for j, seq_name in enumerate(seq):
        if seq_name == "pre":
            images_path = pre_images_path
            mask_path = pre_mask_path
        elif seq_name == "post":
            images_path = post_images_path
            mask_path = post_mask_path
        else:
            raise ValueError("seq_name must be 'pre' or 'post'")
        
        for i in range(start_i, num_difs): # e.g. 0,1,2,3,4   i <= iter_num - 1
            print(f"====i:{i} / {num_difs}====")
            dif_name = difkeys[i]
            print("loading: ", dif_name)
            difmap = difs[dif_name]
            query_points = []
            for k in range(len_xy):
                x, y = xy[k].cpu().numpy()  # 从 GPU 上获取数据
                if difmap[int(y), int(x)]:
                    query_points.append([x, y])
            if query_points == []:
                continue
            points = torch.from_numpy(np.array(query_points)).to(device)  # 移动到 GPU
            num_pts = points.shape[0]
            print("query_points len: ", num_pts) # (n, 2), n is the number of points in the dif map
            point_labels = torch.ones(points.shape[0], 1).to(device) # (n, 1)

            img_name = dif_name.split("_")[0]
            image_path = os.path.join(images_path, img_name + ".png")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = ToTensor()(image).to(device)  # 移动到 GPU
            # label is 1 = positive points, 0 = negative points
            # all 1 will be used to generate masks, 0 will be used to generate background
            try:
                with torch.no_grad():
                    # predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
                    #         img_tensor,
                    #         points.reshape(1, num_pts, 1, 2),
                    #         point_labels.reshape(1, num_pts, 1),
                    #         model,
                    #         device,
                    #     ) # (963, 3, 1200, 1600)
                    predicted_masks, predicted_iou = get_predictions_with_batching(
                        img_tensor,
                        points.reshape(1, num_pts, 1, 2),
                        point_labels.reshape(1, num_pts, 1),
                        model,
                        device,
                        batch_size=50 #100 oom  
                    )
            except Exception as e:
                print(f"Error: OOM, Skip {seq_name} {dif_name} when i = {i}")
                if seq_name == "pre":
                    pre_skip_i.append(i)
                elif seq_name == "post":
                    post_skip_i.append(i)
                continue
            
            if predicted_masks == []:
                continue
            rle = [mask_to_rle_pytorch(m[0:1].to('cpu')) for m in predicted_masks] 
            # predicted mask: (963, 3, 1200, 1600)
            # m: (3, 1200, 1600)
            # m[0:1]: (1, 1200, 1600) the first mask
            # rle: compressed mask
            print("rle shape: ", len(rle)) 
            if rle == []:
                continue
            predicted_masks = process_small_region(rle, device) 
            print("predicted_masks: ", predicted_masks) # list of bool matrix, list of masks
            print("len(predicted_masks): ", len(predicted_masks)) # len = 28
            print("predicted_masks[0].shape: ", predicted_masks[0].shape) # (1200, 1600)
            # return predicted_masks

            if predicted_masks == []:
                continue
            
            mask = predicted_masks
            mask_only_img = np.zeros((mask[0].shape[0], mask[0].shape[1], 4))  # RGBA
            mask_only_img[:,:,3] = 1  # set the alpha channel to 1, not transparent

            if seq_name == "pre":
                concat_mask_pre = np.zeros((mask[0].shape[0], mask[0].shape[1], 1))

                for ann in mask:
                    m = ann
                    color_mask = [1,1,1,1]  # RGBA
                    mask_only_img[m] = color_mask  
                    concat_mask_pre[m] = 1
            
                os.makedirs(pre_mask_path, exist_ok=True)
                plt.imsave(os.path.join(pre_mask_path, dif_name + ".png"), mask_only_img)

                # concat_mask_pre: 0 1, final mask for all dif in pre.

            elif seq_name == "post":
                concat_mask_post = np.zeros((mask[0].shape[0], mask[0].shape[1], 1))

                for ann in mask:
                    m = ann
                    color_mask = [1,1,1,1]
                    mask_only_img[m] = color_mask
                    concat_mask_post[m] = 1

                os.makedirs(post_mask_path, exist_ok=True)
                plt.imsave(os.path.join(post_mask_path, dif_name + ".png"), mask_only_img)

            else:
                raise ValueError("seq_name must be 'pre' or 'post'")


    print("pre_skip_i: ", pre_skip_i)
    print("post_skip_i: ", post_skip_i)