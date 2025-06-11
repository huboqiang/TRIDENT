import sys
import numpy as np
import pandas as pd

sys.path.append('/cluster/home/panfy/software/packages/istar/')
from istar.impute import get_data, normalize, SpotDataset
from istar.utils import read_string

def dataset_sample(sample_name, model_name="conch_v1"):
    prefix = f"/cluster/home/panfy/projects/panlab/analysis/istar/mliti-result/auto_ratio/{sample_name}/{model_name}/"
    embs, cnts, locs = get_data(prefix)

    factor = 16
    radius = int(read_string(f'{prefix}radius.txt'))
    radius = radius / factor

    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    __, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)

    n_train = cnts.shape[0]
    batch_size = min(128, n_train//16)
    x = embs.copy()
    y = cnts
    dataset = SpotDataset(x, y, locs, radius)
    return dataset

if __name__ == "__main__":
    l_ds = []
    for sample in ['NCBI783', 'NCBI785', 'TENX95', 'TENX99']:
        ds = dataset_sample(sample, 'conch_v1')
        l_ds.append(ds)

    print(f"Dataset created with {ds}")

    data_list = []
    for ds in l_ds:
        for i in range(len(ds)):
            x, y = ds[i]
            item = {"x": x, "y": y}
            data_list.append(item)

    # 变成 dataframe
    df = pd.DataFrame(data_list)
    df.to_parquet("debug.parquet")