import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io
import cv2
GRID_SIZE = 64 # 32
import argparse
from efficient_sam.build_efficient_sam import build_efficient_sam_vits
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

        if len(new_masks) == 0:
            return []

        # Recalculate boxes and remove any new duplicates # Here! 963 to 28.
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

# def get_predictions_given_embeddings_and_queries_batchified(img, points, point_labels, model):
#     predicted_masks, predicted_iou = model(img[None, ...], points, point_labels)
#     return predicted_masks, predicted_iou

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
    0. prepare: dif_images(denoised & segmented dif map), difs.npz, pre_images, post_images (pre_normal_images & post_selected_images). all name matched. (test_preprocess_difmap.py)

    1. combine many image to a batch (give up)
    2. for each dif in difs.npz(~300), get a enabled grid whose point in dif map has 'True' value. save the query point list.
    3. for each group of query point, run everything ours and get the mask list. (test_segment_difmap.py)
    NOTE don't forget disable the image embedding 'return'

    Use the IoU between difmap and seg mask to decide in&out, and get one confident mask for each dif.
    4. compute mask list and dif's IoU and get the best mask for each dif. btw threshold to 0.3.
    5. for each dif in json, if the seq not null, follow the seq to select mask. in-post, out-pre.

    Semantically discard the fake change.
    Now we have out-pre masks and in-post masks.
    6. follow the masks name to find out images, store the in-pre in-post out-pre out-post masked images.
    7. take out as example, calculate the cos similarity between out-pre and out-post 
    NOTE This step i modify forward function!
    NOTE the masked image between pre and post are tooo similar...
    
    Semantic clustering
    7. for each in or out masked image, get the semantic feature (image encoder embedding) and calculate the cosine similarity, store the neighbour matrix. and store the mask class k.
    NOTE This step i modify forward function! 


    A new pipeline:
    -1. preprepare: image embedding difmap (test img-embd-for-difmap.py)

    '''

    # !git clone https://github.com/yformer/EfficientSAM.git
    # import os
    # os.chdir("EfficientSAM")

    # Todo
    # parser = argparse.ArgumentParser()

    batchsize = 4
    difs_npz_path = "data/denoise_dif_map/003/difs.npz"
    pre_images_path = "data/denoise_dif_map/003/pre_normal_images"
    post_images_path = "data/denoise_dif_map/003/post_selected_images"
    pre_mask_path = "data/denoise_dif_map/003/pre_masks"
    post_mask_path = "data/denoise_dif_map/003/post_masks"
    original_image_w = 1600
    original_image_h = 1200
    start_i = 0 # 0
    seq = ["pre","post"]
    # seq = ["pre"]

    
    # with zipfile.ZipFile("weights/efficient_sam_vits.pt.zip", 'r') as zip_ref:
    #     zip_ref.extractall("weights")
    efficient_sam_vits_model = build_efficient_sam_vits()
    efficient_sam_vits_model.eval()
    model = efficient_sam_vits_model.cpu()

    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * original_image_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * original_image_h
            xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy))
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
                x, y = xy[k]
                if difmap[int(y), int(x)]:
                    query_points.append([x, y])
            if query_points == []:
                continue
            points = torch.from_numpy(np.array(query_points))
            num_pts = points.shape[0]
            print("query_points len: ", num_pts) # (n, 2), n is the number of points in the dif map
            point_labels = torch.ones(points.shape[0], 1) # (n, 1)

            img_name = dif_name.split("_")[0]
            image_path = os.path.join(images_path, img_name + ".png")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = ToTensor()(image)
            # label is 1 = positive points, 0 = negative points
            # all 1 will be used to generate masks, 0 will be used to generate background
            with torch.no_grad():
                predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
                    img_tensor.cpu(),
                    points.reshape(1, num_pts, 1, 2).cpu(),
                    point_labels.reshape(1, num_pts, 1).cpu(),
                    model.cpu(),
                ) # (963, 3, 1200, 1600)
            if predicted_masks == []:
                continue
            rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks] 
            # predicted mask: (963, 3, 1200, 1600)
            # m: (3, 1200, 1600)
            # m[0:1]: (1, 1200, 1600) the first mask
            # rle: compressed mask
            print("rle shape: ", len(rle)) 
            if rle == []:
                continue
            predicted_masks = process_small_region(rle) 
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


            # 因为中间会断，所以暂时先不把它存成npz格式了。否则要么全用cpu特别慢，要么全用gpu还是会断，npz断点续传好像还得研究一下。其实也可以。。。直接找key没有就append就可以了，只不过最后要sort一下。




# deprecated bachify
        # temperarily give up to batchify it... compute resource is limited; the query points num are not aligned.
        # start should be included but end shouldn't be included!!!!! the index of difs begin at 0!
        # start = batchsize * i # e.g. 0,4,8,12,16
        # end = batchsize * (i + 1) if i != iter_num - 1 else num_difs # e.g. 4,8,12,16,17

        # batch_images = []
        # batch_points = []
        # batch_labels = []

        # for j in range(start, end):
        #     dif_name = difkeys[j]
        #     print("loading: ", dif_name)
        #     difmap = difs[dif_name]
        #     quary_points = []
        #     for k in range(len_xy):
        #         x, y = xy[k]
        #         if difmap[int(y), int(x)]:
        #             quary_points.append([x, y])
        #     query_points = torch.from_numpy(np.array(quary_points))
        #     print("query_points.shape: ", query_points.shape) # (n, 2), n is the number of points in the dif map
        #     query_label = torch.ones(query_points.shape[0], 1) # (n, 1)
        #     batch_points.append(query_points)
        #     batch_labels.append(query_label)

        #     img_name = dif_name.split("_")[0]
        #     image_path = os.path.join(pre_images_path, img_name + ".png")
        #     image = cv2.imread(image_path)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     img_tensor = ToTensor()(image)
        #     batch_images.append(img_tensor)

        # images = torch.stack(batch_images, dim=0)
        # points = torch.stack(batch_points, dim=0) # NOT ALIGN
        # points_labels = torch.stack(batch_labels, dim=0)
        # num_pts = points.shape[1]
        # print("images.shape: ", images.shape) 




# def run_everything_ours(img_path, model):
    # model = model.cpu()
    # image = cv2.imread(image_path)
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_tensor = ToTensor()(image)
    # _, original_image_h, original_image_w = img_tensor.shape

    # # HERE it is actually evenly spaced points
    # xy = []
    # for i in range(GRID_SIZE):
    #     curr_x = 0.5 + i / GRID_SIZE * original_image_w
    #     for j in range(GRID_SIZE):
    #         curr_y = 0.5 + j / GRID_SIZE * original_image_h
    #         xy.append([curr_x, curr_y])
    # xy = torch.from_numpy(np.array(xy))
    # print("xy.shape: ", xy.shape) # (1024, 2) if GRID_SIZE = 32

    # points = xy
    # num_pts = xy.shape[0]
    # point_labels = torch.ones(num_pts, 1) # (1024, 1)
    # # label is 1 = positive points, 0 = negative points
    # # all 1 will be used to generate masks, 0 will be used to generate background
    # with torch.no_grad():
    # #   predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
    # #           img_tensor.cpu(),
    # #           points.reshape(1, num_pts, 1, 2).cpu(),
    # #           point_labels.reshape(1, num_pts, 1).cpu(),
    # #           model.cpu(),
    # #       ) # (963, 3, 1200, 1600)
    #   predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries_batchified(
    #           img_tensor.cpu(),
    #           points.reshape(1, num_pts, 1, 2).cpu(),
    #           point_labels.reshape(1, num_pts, 1).cpu(),
    #           model.cpu(),
    #       ) # (963, 3, 1200, 1600)
    # rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks] 
    # # predicted mask: (963, 3, 1200, 1600)
    # # m: (3, 1200, 1600)
    # # m[0:1]: (1, 1200, 1600) the first mask
    # # rle: compressed mask
    # print("rle shape: ", len(rle)) 
    # predicted_masks = process_small_region(rle) 
    # print("predicted_masks: ", predicted_masks) # list of bool matrix, list of masks
    # print("len(predicted_masks): ", len(predicted_masks)) # len = 28
    # print("predicted_masks[0].shape: ", predicted_masks[0].shape) # (1200, 1600)
    # return predicted_masks





# the figure showing in main

    # fig, ax = plt.subplots(1, 2, figsize=(30, 30))
    # image_path = "/home/stu7/projects/EfficientSAM/data/test_imgs/00003.png"
    # image = np.array(Image.open(image_path))
    # ax[0].imshow(image)
    # ax[0].title.set_text("Original")
    # ax[0].axis('off')

    # ax[1].imshow(image)
    # mask_efficient_sam_vits = run_everything_ours(image_path, efficient_sam_vits_model) # list of bool matrix
    # show_anns_ours(mask_efficient_sam_vits, ax[1])
    # ax[1].title.set_text("EfficientSAM")
    # ax[1].axis('off')
    # # plt.show()
    # plt.savefig("data/test_imgs/efficient_sam.png")




    # mask = predicted_masks
    #     # fig, ax = plt.subplots(figsize=(15, 15)) 
    #     mask_only_img = np.zeros((mask[0].shape[0], mask[0].shape[1], 4))  # RGBA

    #     for ann in mask:
    #         m = ann
    #         color_mask = np.concatenate([np.random.random(3), [0.5]])  # RGBA
    #         mask_only_img[m] = color_mask  

    #     # ax.imshow(mask_only_img) 
    #     # ax.axis('off')  
    #     # ax.title.set_text("Mask Only") 
    #     os.makedirs(pre_mask_path, exist_ok=True)
    #     # save image
    #     plt.imsave(os.path.join(pre_mask_path, dif_name + ".png"), mask_only_img)

    #     # plt.savefig(os.path.join(pre_mask_path, dif_name + ".png"))