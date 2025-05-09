
# # EfficientSAM environment
# # copy pre gt and post renders
# # NOTE the same size!!!!! splited!!! just copy the gt!!!
# # maybe i can combine the following scripts
# # ps, considered that i have to modify the efficientsam scripts, i should directly copied that project here...?

# # get difference map
# # XXX modify the 'efficient_sam.py'. change the returns
# python test_img-embd-for-difmap.py --dir1 data/denoise_dif_map/005_lego/pre_gt --dir2 data/denoise_dif_map/005_lego/post_renders --heatmap data/denoise_dif_map/005_lego/heatmap --difmap data/denoise_dif_map/005_lego/difmap --threshold 0.27

# # python test_img-embd-for-difmap.py --dir1 data/denoise_dif_map/006_desk/pre_gt --dir2 data/denoise_dif_map/006_desk/post_renders --heatmap data/denoise_dif_map/006_desk/heatmap --difmap data/denoise_dif_map/006_desk/difmap --threshold 0.27



# # denoise and segment difference map 
# python test_preprocess_difmap.py --input_directory data/denoise_dif_map/005_lego/difmap --output_directory data/denoise_dif_map/005_lego/dif_images --mask_output_file data/denoise_dif_map/005_lego/difs.npz

# # python test_preprocess_difmap.py --input_directory data/denoise_dif_map/006_desk/difmap --output_directory data/denoise_dif_map/006_desk/dif_images --mask_output_file data/denoise_dif_map/006_desk/difs.npz


# # prompt difference map for precise pre and post mask
# # XXX NOTE in efficientsam.py, change the returns
# # time consuming step
# # if break, use the none gpu version
# python test_segment_difmap_gpu.py --difs_npz_path data/denoise_dif_map/005_lego/difs.npz --pre_images_path data/denoise_dif_map/005_lego/pre_gt --post_images_path data/denoise_dif_map/005_lego/post_renders --pre_mask_path data/denoise_dif_map/005_lego/pre_masks --post_mask_path data/denoise_dif_map/005_lego/post_masks

# # python test_segment_difmap_gpu.py --difs_npz_path data/denoise_dif_map/006_desk/difs.npz --pre_images_path data/denoise_dif_map/006_desk/pre_gt --post_images_path data/denoise_dif_map/006_desk/post_renders --pre_mask_path data/denoise_dif_map/006_desk/pre_masks --post_mask_path data/denoise_dif_map/006_desk/post_masks 


# # use pre & post mask to calculate in and out mask
# python test_choose_max_IoU.py --root_dir data/denoise_dif_map/005_lego

# # python test_choose_max_IoU.py --root_dir data/denoise_dif_map/006_desk







#!/bin/bash

# 设置数据集根目录
ROOT_DIR="data/denoise_dif_map"  # 数据集根目录
# OUTPUT_ROOT="output"             # 输出根目录

# 数据集名称列表和对应的编号列表
# DATASETS=("card" "sculpture" "mural" "pottery" "text" "potting") 
DATASETS=("livingroom-007") 
DATASET_NUMS=("013")  # 数据集编号
RENDER_POSTCAM_VALUES=("false")  


# 执行每个数据集的处理流程
for i in "${!DATASETS[@]}"; do
    NAME="${DATASETS[$i]}"
    DATASET_NUM="${DATASET_NUMS[$i]}"
    DATASET_NAME="${DATASET_NUM}_${NAME}"
    RENDER_POSTCAM_VALUE="${RENDER_POSTCAM_VALUES[$i]}"

    # -------------------------
    # 获取差异图
    # -------------------------
    # if [ "${RENDER_POSTCAM_VALUE}" == "true" ]; then
    #     echo "RENDER_POSTCAM_VALUE true"
    #     python test_img-embd-for-difmap.py --dir1 ${ROOT_DIR}/${DATASET_NAME}/train-post --dir2 ${ROOT_DIR}/${DATASET_NAME}/renders --heatmap ${ROOT_DIR}/${DATASET_NAME}/heatmap --difmap ${ROOT_DIR}/${DATASET_NAME}/difmap --threshold 0.27
    # else
    #     echo "RENDER_POSTCAM_VALUE false"
    #     python test_img-embd-for-difmap.py --dir1 ${ROOT_DIR}/${DATASET_NAME}/train-pre --dir2 ${ROOT_DIR}/${DATASET_NAME}/renders --heatmap ${ROOT_DIR}/${DATASET_NAME}/heatmap --difmap ${ROOT_DIR}/${DATASET_NAME}/difmap --threshold 0.27
    # fi

    python test_img-embd-for-difmap.py --dir1 ${ROOT_DIR}/${DATASET_NAME}/gt --dir2 ${ROOT_DIR}/${DATASET_NAME}/renders --heatmap ${ROOT_DIR}/${DATASET_NAME}/heatmap --difmap ${ROOT_DIR}/${DATASET_NAME}/difmap --threshold 0.27
    

    # -------------------------
    # 去噪并分割差异图
    # -------------------------
    python test_preprocess_difmap.py --input_directory ${ROOT_DIR}/${DATASET_NAME}/difmap --output_directory ${ROOT_DIR}/${DATASET_NAME}/dif_images --mask_output_file ${ROOT_DIR}/${DATASET_NAME}/difs.npz

    # -------------------------
    # 使用差异图prompt efficientsam 生成前后masks
    # -------------------------
    python test_segment_difmap_gpu.py --difs_npz_path ${ROOT_DIR}/${DATASET_NAME}/difs.npz --pre_images_path ${ROOT_DIR}/${DATASET_NAME}/gt --post_images_path ${ROOT_DIR}/${DATASET_NAME}/renders --pre_mask_path ${ROOT_DIR}/${DATASET_NAME}/pre_masks --post_mask_path ${ROOT_DIR}/${DATASET_NAME}/post_masks

    # -------------------------
    # 使用前后掩码计算 in 和 out 掩码
    # -------------------------
    python test_choose_max_IoU.py --root_dir ${ROOT_DIR}/${DATASET_NAME}

done
