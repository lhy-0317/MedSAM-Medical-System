# # -*- coding: utf-8 -*-
# """
# Evaluation script for MedSAM model
# Computes DSC (Dice Similarity Coefficient) and NSD (Normalized Surface Dice)
# and saves segmentation visualizations (no plt.show)
# """

# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage as ndi
# from tqdm import tqdm  # ✅ 新增进度条
# from segment_anything import sam_model_registry

# # 从你之前的训练脚本里导入以下三个元素
# from train_one_gpu import NpyDataset, MedSAM, show_mask, show_box  # ← 修改为你的训练文件名

# # ======================= Metric functions =======================
# def dice_score(gt, pred):
#     gt = gt.astype(bool)
#     pred = pred.astype(bool)
#     intersection = np.logical_and(gt, pred).sum()
#     return 2.0 * intersection / (gt.sum() + pred.sum() + 1e-8)

# def compute_surface(mask):
#     eroded = ndi.binary_erosion(mask)
#     surface = mask ^ eroded
#     return surface

# def normalized_surface_dice(gt, pred, tolerance=2):
#     gt, pred = gt.astype(bool), pred.astype(bool)
#     surface_gt = compute_surface(gt)
#     surface_pred = compute_surface(pred)

#     dist_gt = ndi.distance_transform_edt(~gt)
#     dist_pred = ndi.distance_transform_edt(~pred)

#     dist_to_pred = dist_pred[surface_gt]
#     dist_to_gt = dist_gt[surface_pred]

#     tp_gt = np.sum(dist_to_pred <= tolerance)
#     tp_pred = np.sum(dist_to_gt <= tolerance)

#     nsd = (tp_gt + tp_pred) / (surface_gt.sum() + surface_pred.sum() + 1e-8)
#     return nsd


# # ======================= Visualization & Evaluation =======================
# def evaluate_and_save(model, dataset, device, save_dir, max_save=10, tolerance=2, desc="Evaluating"):
#     os.makedirs(save_dir, exist_ok=True)
#     model.eval()

#     dice_scores, nsd_scores = [], []

#     # 使用 tqdm 显示进度
#     with torch.no_grad():
#         for i in tqdm(range(len(dataset)), desc=desc, ncols=100):
#             img, gt, bbox, name = dataset[i]

#             img_input = img.unsqueeze(0).to(device)
#             bbox_np = bbox[None, :].numpy()

#             pred = model(img_input, bbox_np)
#             pred_sig = torch.sigmoid(pred).cpu().squeeze().numpy()
#             pred_bin = (pred_sig > 0.5).astype(np.uint8)
#             gt_np = gt.squeeze().numpy().astype(np.uint8)

#             dice = dice_score(gt_np, pred_bin)
#             nsd = normalized_surface_dice(gt_np, pred_bin, tolerance)
#             dice_scores.append(dice)
#             nsd_scores.append(nsd)

#             # 保存前几张图像用于可视化
#             if i < max_save:
#                 fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#                 ax[0].imshow(np.transpose(img.numpy(), (1, 2, 0)))
#                 ax[0].set_title("Input Image")
#                 ax[0].axis("off")

#                 ax[1].imshow(gt_np, cmap="gray")
#                 ax[1].set_title("Ground Truth")
#                 ax[1].axis("off")

#                 ax[2].imshow(np.transpose(img.numpy(), (1, 2, 0)))
#                 show_mask(pred_bin, ax[2])
#                 show_box(bbox.numpy(), ax[2])
#                 ax[2].set_title("Prediction")
#                 ax[2].axis("off")

#                 plt.subplots_adjust(wspace=0.05, hspace=0)
#                 save_path = os.path.join(save_dir, f"{name.replace('.npy', '.png')}")
#                 plt.savefig(save_path, bbox_inches="tight", dpi=200)
#                 plt.close()

#     mean_dice = np.mean(dice_scores)
#     mean_nsd = np.mean(nsd_scores)
#     return mean_dice, mean_nsd


# # ======================= Main Entry =======================
# if __name__ == "__main__":
#     # ------------ 参数设置 ------------
#     device = "cuda:0"
#     dataset_path = "data/npy/CT_Abd"
#     checkpoint_sam = "work_dir/SAM/sam_vit_b_01ec64.pth"
#     model_ckpt_path = "/home/lyx/MedSAM/work_dir/MedSAM/medsam_vit_b.pth"
#     save_dir_train = "eval_results_train_medsam"
#     save_dir_test = "eval_results_test_medsam"  # 改为测试集结果文件夹
#     tolerance = 2  # NSD 容忍距离
#     test_num = 922  # 后 922 个样本作为测试集

#     # ------------ 模型加载 ------------
#     sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_sam)
#     model = MedSAM(
#         sam_model.image_encoder,
#         sam_model.mask_decoder,
#         sam_model.prompt_encoder
#     ).to(device)

#     ckpt = torch.load(model_ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model"])
#     print("✅ Model loaded successfully.")

#     # ------------ 数据划分 ------------
#     all_dataset = NpyDataset(dataset_path)
#     n = len(all_dataset)
#     if n <= test_num:
#         raise ValueError(f"❌ 数据集样本数不足，当前共 {n} 个样本，无法划分出 {test_num} 个测试样本。")

#     n_train = n - test_num
#     indices = list(range(n))
#     train_indices = indices[:n_train]
#     test_indices = indices[n_train:]

#     train_dataset = torch.utils.data.Subset(all_dataset, train_indices)
#     test_dataset = torch.utils.data.Subset(all_dataset, test_indices)

#     print(f"Dataset size: {n}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")

#     # ------------ 训练集评估 ------------
#     print("\n🚀 Evaluating on training set...")
#     dice_tr, nsd_tr = evaluate_and_save(
#         model, train_dataset, device, save_dir_train,
#         max_save=10, tolerance=tolerance, desc="Train Set"
#     )

#     # ------------ 测试集评估 ------------
#     print("\n🧪 Evaluating on test set...")
#     dice_test, nsd_test = evaluate_and_save(
#         model, test_dataset, device, save_dir_test,
#         max_save=10, tolerance=tolerance, desc="Test Set"
#     )

#     # ------------ 打印并保存结果 ------------
#     print(f"\n📊 Results Summary:")
#     print(f"Train → Dice={dice_tr:.4f}, NSD={nsd_tr:.4f}")
#     print(f"Test  → Dice={dice_test:.4f}, NSD={nsd_test:.4f}")

#     result_file = "final_metrics_medsam.txt"
#     with open(result_file, "w") as f:
#         f.write(f"Train Dice: {dice_tr:.4f}, Train NSD: {nsd_tr:.4f}\n")
#         f.write(f"Test Dice: {dice_test:.4f}, Test NSD: {nsd_test:.4f}\n")
#     print(f"✅ Results saved to {result_file}")


# -*- coding: utf-8 -*-
"""
Evaluation script for MedSAM
Computes Dice Similarity Coefficient (DSC) and Normalized Surface Dice (NSD)
and saves segmentation visualizations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from tqdm import tqdm
from segment_anything import sam_model_registry
from skimage import transform

# -----------------------------
# 从训练脚本导入数据集和可视化函数
# -----------------------------
from train_one_gpu import NpyDataset, show_mask, show_box  # 修改为你的脚本路径

# -----------------------------
# Metric functions
# -----------------------------
def dice_score(gt, pred):
    gt, pred = gt.astype(bool), pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    return 2.0 * inter / (gt.sum() + pred.sum() + 1e-8)

def compute_surface(mask):
    eroded = ndi.binary_erosion(mask)
    return mask ^ eroded

def normalized_surface_dice(gt, pred, tolerance=2):
    gt, pred = gt.astype(bool), pred.astype(bool)
    surface_gt, surface_pred = compute_surface(gt), compute_surface(pred)
    dist_gt, dist_pred = ndi.distance_transform_edt(~gt), ndi.distance_transform_edt(~pred)
    tp_gt = np.sum(dist_pred[surface_gt] <= tolerance)
    tp_pred = np.sum(dist_gt[surface_pred] <= tolerance)
    return (tp_gt + tp_pred) / (surface_gt.sum() + surface_pred.sum() + 1e-8)

# -----------------------------
# 推理函数（与 MedSAM_Inference.py 一致）
# -----------------------------
@torch.no_grad()
def medsam_inference(model, img_embed, box_1024, H, W):
    import torch.nn.functional as F
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    medsam_seg = (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return medsam_seg

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_and_save(model, dataset, device, save_dir, max_save=10, tolerance=2, desc="Evaluating"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    dice_scores, nsd_scores = [], []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc, ncols=100):
            img_tensor, gt_tensor, bbox, name = dataset[i]
            img_np = np.transpose(img_tensor.numpy(), (1, 2, 0))
            H, W, _ = img_np.shape

            # normalize & scale to 1024x1024
            img_1024 = transform.resize(img_np, (1024, 1024), order=3, preserve_range=True)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

            # compute embedding
            img_embed = model.image_encoder(img_1024_tensor)

            box_np = bbox[None, :].numpy()
            box_1024 = box_np / np.array([W, H, W, H]) * 1024

            pred_mask = medsam_inference(model, img_embed, box_1024, H, W)
            gt_np = gt_tensor.squeeze().numpy().astype(np.uint8)

            dice = dice_score(gt_np, pred_mask)
            nsd = normalized_surface_dice(gt_np, pred_mask, tolerance)
            dice_scores.append(dice)
            nsd_scores.append(nsd)

            if i < max_save:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(img_np)
                ax[0].set_title("Input Image"); ax[0].axis("off")

                ax[1].imshow(gt_np, cmap="gray")
                ax[1].set_title("Ground Truth"); ax[1].axis("off")

                ax[2].imshow(img_np)
                show_mask(pred_mask, ax[2])
                show_box(bbox.numpy(), ax[2])
                ax[2].set_title("Prediction"); ax[2].axis("off")

                plt.savefig(os.path.join(save_dir, f"{name.replace('.npy', '.png')}"),
                            bbox_inches="tight", dpi=200)
                plt.close()

    return np.mean(dice_scores), np.mean(nsd_scores)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    device = "cuda:0"
    dataset_path = "data/npy/CT_Abd"
    model_ckpt_path = "work_dir/MedSAM/medsam_vit_b.pth"
    save_dir_train = "eval_results_train_medsam"
    save_dir_test = "eval_results_test_medsam"
    tolerance = 2
    test_num = 922

    # Load trained MedSAM
    print("🔄 Loading MedSAM model directly...")
    model = sam_model_registry["vit_b"](checkpoint=model_ckpt_path)
    model = model.to(device)
    model.eval()
    print("✅ MedSAM model loaded successfully.")

    # Dataset split
    all_dataset = NpyDataset(dataset_path)
    n = len(all_dataset)
    n_train = n - test_num
    train_dataset = torch.utils.data.Subset(all_dataset, list(range(0, n_train)))
    test_dataset = torch.utils.data.Subset(all_dataset, list(range(n_train, n)))
    print(f"Dataset size: {n}, Train: {n_train}, Test: {n - n_train}")

    # Train set eval
    print("\n🚀 Evaluating on training set...")
    dice_tr, nsd_tr = evaluate_and_save(model, train_dataset, device, save_dir_train, max_save=10, tolerance=tolerance, desc="Train Set")

    # Test set eval
    print("\n🧪 Evaluating on test set...")
    dice_te, nsd_te = evaluate_and_save(model, test_dataset, device, save_dir_test, max_save=10, tolerance=tolerance, desc="Test Set")

    print(f"\n📊 Results Summary:\nTrain → Dice={dice_tr:.4f}, NSD={nsd_tr:.4f}\nTest  → Dice={dice_te:.4f}, NSD={nsd_te:.4f}")
    with open("final_metrics_medsam.txt", "w") as f:
        f.write(f"Train Dice: {dice_tr:.4f}, Train NSD: {nsd_tr:.4f}\n")
        f.write(f"Test Dice: {dice_te:.4f}, Test NSD: {nsd_te:.4f}\n")
    print("✅ Results saved to final_metrics_medsam.txt")