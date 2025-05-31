
from typing import List, Union
import scanpy as sc
import torch
import torch.nn.functional as F
import numpy as np
import pickle

import matplotlib.pyplot as plt
def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def save_pickle(x, filename):
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


Type_NCHW = torch.FloatTensor
def align_tensor_to_mask(tensor_NCHW: Type_NCHW, mask: np.ndarray, ratio: float, offset: int = 16):
    """
    Aligns a list of tensors to a given mask using a specified ratio.
    
    Parameters:
        tensor_NCHW (torch.Tensor): The tensor to align.
        mask (np.ndarray): The mask to align the tensors to. [H, W, ...]
        ratio (float): The ratio for alignment.
        
    Returns:
        List[torch.Tensor]: List of aligned tensors.
    """    
    resized_tensor = F.interpolate(tensor_NCHW, scale_factor=ratio, mode='bilinear', align_corners=True)
    h_, w_ = resized_tensor.shape[2:]

    h, w = mask.shape[0:2]
    pad_top = offset
    pad_bottom = max((h - h_ - offset), 0)
    pad_left = offset
    pad_right = max((w - w_ - offset), 0)

    aligned_tensors = F.pad(resized_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)[:, :, 0:h, 0:w]
    
    return aligned_tensors


def adata_to_tensor_tile(adata: sc.AnnData) -> torch.Tensor:
    # 1. 读取并取整空间坐标
    spatial = adata.obsm["spatial"].astype(int)
    x_coords = spatial[:, 0]
    y_coords = spatial[:, 1]

    # 2. 推断水平和垂直间隔
    #    取唯一值、排序，然后求最小非零差值
    ux = np.unique(x_coords)
    uy = np.unique(y_coords)
    dx = np.diff(ux)
    dy = np.diff(uy)
    interval_x = dx[dx > 0].min()
    interval_y = dy[dy > 0].min()
    if interval_x != interval_y:
        raise ValueError("patches are not square")
    interval=interval_x

    # 3. 将 coords 归一到 (0,0) 起点，并转换为网格索引
    min_x, min_y = x_coords.min(), y_coords.min()
    x_grid = (x_coords - min_x) // interval
    y_grid = (y_coords - min_y) // interval

    # 4. 计算网格尺寸 H', W'
    Wp = int(x_grid.max()) + 1
    Hp = int(y_grid.max()) + 1

    # 5. 准备数据矩阵 (n_obs, C)
    #    如果是稀疏矩阵，则先转为稠密
    X = adata.X
    data_mat = np.asarray(X)

    C = data_mat.shape[1]

    # 6. 初始化 (C, H', W') 零数组，并按网格索引填充值
    arr = np.zeros((C, Hp, Wp), dtype=data_mat.dtype)
    for i in range(spatial.shape[0]):
        gx, gy = x_grid[i], y_grid[i]
        arr[:, gy, gx] = data_mat[i, :]

    # 7. 转为 torch.Tensor 并增加 batch 维
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, C, Hp, Wp)
    return tensor



# s1 = !cat /cluster/home/panfy/projects/panlab/analysis/istar/pixel-size.txt
# s2 = !cat /cluster/home/panfy/projects/panlab/analysis/istar/pixel-size-raw.txt


def run_istar(dir_istar: str, model_name="virchow"):
    with open(f"{dir_istar}/pixel-size.txt") as f:
        v1 = float( f.readlines()[0] )

    with open(f"{dir_istar}/pixel-size-raw.txt") as f:
        v2 = float( f.readlines()[0] )

    ratio = v2 / v1
    if model_name == "virchow":
        ratio = v2 / v1 * 14/16

    adata_cls =  sc.read_h5ad(f"{dir_istar}/40x_224px_0px_overlap/features_{model_name}/he-raw.h5ad")
    adata_sub = sc.read_h5ad(f"{dir_istar}/40x_224px_0px_overlap/attention_features_{model_name}/he-raw_flatten.h5ad")

    tensor_sub = adata_to_tensor_tile(adata_sub)
    N, C, H, W = tensor_sub.shape
    tensor_cls = F.interpolate(adata_to_tensor_tile(adata_cls), size=[H, W], mode="bilinear", align_corners=True)


    raw_emb_hist = load_pickle(f"{dir_istar}/defult_embeddings-hist.pickle")
    tensor_cls_resized = align_tensor_to_mask(tensor_cls, raw_emb_hist['cls'][0], ratio=ratio, offset=16)
    tensor_sub_resized = align_tensor_to_mask(tensor_sub, raw_emb_hist['sub'][0], ratio=ratio, offset=16)

    emb_hist = {}
    emb_hist["rgb"] = raw_emb_hist["rgb"]
    emb_hist["cls"] = [ tensor_cls_resized[0, i_channel].numpy() for i_channel in range(tensor_cls_resized.shape[1]) ]
    emb_hist["sub"] = [ tensor_sub_resized[0, i_channel].numpy() for i_channel in range(tensor_sub_resized.shape[1]) ]


    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121)
    img = np.stack(raw_emb_hist["rgb"], axis=2)
    ax1.imshow(img)

    img2_ = img.copy()
    img2_[:, :, 0:2] = 100*tensor_sub_resized[0, 0:2].permute(1, 2, 0)
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2_)
    fig.savefig(f"./embeddings-hist-{model_name}.png")
    save_pickle(emb_hist, f"./embeddings-hist-{model_name}.pickle")

if __name__ == "__main__":
    run_istar("/cluster/home/panfy/projects/panlab/analysis/istar", model_name="virchow")
    # run_istar()