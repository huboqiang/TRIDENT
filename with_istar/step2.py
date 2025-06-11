import numpy as np
import pandas as pd
import torch, json, os
from scipy import sparse
from anndata import AnnData
import torch.nn.functional as F
import sys, math
import scanpy as sc
sys.path.append('/cluster/home/panfy/software/packages')
sys.path.append('/cluster/home/panfy/projects')
from with_istar.utils_istar import load_pickle, save_pickle, join, load_image
from einops import rearrange, reduce, repeat
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
import geopandas as gpd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import argparse
from math import floor
import cv2

def adata_to_tensor_tile(adata: AnnData) -> torch.Tensor:
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
    if sparse.issparse(X):
        data_mat = X.toarray()
    else:
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

def align_tensor(tensor,img_array,ratio=0):
    if ratio==0:
        ratio=floor(min(img_array.shape[0]//16/tensor.shape[2],img_array.shape[1]//16/tensor.shape[3])*100)/100
    resized_tensor = F.interpolate(tensor, scale_factor=ratio, mode='bilinear', align_corners=True)
    h_, w_ = resized_tensor.shape[2:]
    # pad left-top offset
    offset = 16
    h=int(img_array.shape[0]/16)
    w=int(img_array.shape[1]/16)
    pad_top = offset
    pad_bottom = int(max((h - h_ - offset), 0))
    pad_left = offset
    pad_right = int(max((w - w_ - offset), 0))
    aligned_tensors = F.pad(resized_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)[:, :, 0:h, 0:w]
    return aligned_tensors

def resize_pillow(img_array, fold=16):
    img = Image.fromarray(img_array)
    # 使用LANCZOS重采样保持质量
    resized = img.resize(
        (img.width//fold, img.height//fold), 
        resample=Image.LANCZOS
    )
    transform = transforms.ToTensor()
    return transform(resized)

def generate_and_visualize_mask(gdf, width, height):
    shapes = ((geom, 255) for geom in gdf.geometry)  # 白值为255
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=rasterio.Affine.identity(),
        fill=0, 
        dtype=np.uint8
    )
    return mask

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--sample', type=str, default=None)
    parser.add_argument('--module', type=str, default=None)
    parser.add_argument('--pixel', type=float, default=None)
    parser.add_argument('--patch', type=int, default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    os.chdir(args.prefix)

    meg=2
    hipt_small_side=16
    target_pixel=0.5

    adata_sub=sc.read(f'/cluster/home/panfy/projects/panlab/analysis/istar/HD_module_extracted_feature/{args.sample}/{args.module}/20x_{args.patch}px_0px_overlap/attention_features_{args.module}/{args.sample}_flatten.h5ad')
    adata_cls=sc.read(f'/cluster/home/panfy/projects/panlab/analysis/istar/HD_module_extracted_feature/{args.sample}/{args.module}/20x_{args.patch}px_0px_overlap/features_{args.module}/{args.sample}.h5ad')
    tensor_sub = adata_to_tensor_tile(adata_sub)
    N, C, H, W = tensor_sub.shape
    tensor_cls = F.interpolate(adata_to_tensor_tile(adata_cls), size=[H, W], mode="bilinear", align_corners=True)
    img = load_image('he.jpg')
    img_array = np.array(img)
    cx=adata_sub.obsm['spatial'][:,0]
    cy=adata_sub.obsm['spatial'][:,1]
    unique_nums = sorted(list(set(cx)))
    sidex=unique_nums[1]-unique_nums[0]
    unique_nums = sorted(list(set(cy)))
    sidey=unique_nums[1]-unique_nums[0]
    if sidex!=sidey:
        raise ValueError("patches are not square")
    #side=sidex
    # if args.pixel<0.5:
    #     ratio=args.pixel/target_pixel*side/hipt_small_side
    # else:
    #     ratio=0.264/target_pixel*side/hipt_small_side
    side=38
    tensor_cls_p=align_tensor(tensor_cls,img_array)
    tensor_sub_p=align_tensor(tensor_sub,img_array)
    dicts={}
    tensor = tensor_cls_p.squeeze(0)
    dicts['cls'] = torch.unbind(tensor, dim=0)
    tensor = tensor_sub_p.squeeze(0)
    dicts['sub'] = torch.unbind(tensor, dim=0)
    dicts['rgb'] = torch.unbind(resize_pillow(img_array), dim=0)
    embed_virchow = {}
    for k in dicts:
        print(k, len(dicts[k]), dicts[k][0].shape, dicts[k][0].dtype)
        embed_virchow[k] = [dicts[k][i].cpu().detach().numpy() for i in range(len(dicts[k]))]

    save_pickle(embed_virchow, "embeddings-hist.pickle")

    gdf=gpd.read_file(f'/cluster/home/panfy/projects/panlab/analysis/istar/HD_module_extracted_feature/{args.sample}/{args.module}/contours_geojson/{args.sample}.geojson')
    imgwhole=load_image(f'/cluster/home/panfy/projects/panlab/analysis/istar/he_pic/{args.sample}.tif')
    mask=generate_and_visualize_mask(gdf,imgwhole.shape[1],imgwhole.shape[0])
    mask=resize_pillow(mask,fold=32)
    mask=mask.unsqueeze(0)
    masks=align_tensor(mask,img_array)
    mask=mask.squeeze(0).squeeze(0)
    mask=(mask >= 0.5).float()
    mask=(mask.numpy() * 255).astype(np.uint8)
    Image.fromarray(mask).save('mask-small.png')

    df = pd.read_csv('locs.tsv', sep='\t')
    cnts = pd.read_csv('cnts.tsv', sep='\t')
    spotkeep = df.loc[(df['x'] > side*16) & (df['x'] < img_array.shape[1]-side*16) & (df['y'] > side*16) & (df['y'] < img_array.shape[0]-side*16), 'spot']
    df = df[df['spot'].isin(spotkeep)]
    cnts = cnts[cnts['spot'].isin(spotkeep)]
    df.to_csv('locs.tsv', sep='\t', index=False)
    cnts.to_csv('cnts.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()