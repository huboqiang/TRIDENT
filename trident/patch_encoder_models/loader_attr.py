import numpy as np
import h5py


from trident.patch_encoder_models.load import Conchv1InferenceEncoder, Conchv15InferenceEncoder, UNIInferenceEncoder, GigaPathInferenceEncoder, VirchowInferenceEncoder

"""
This file contains an assortment of pretrained patch encoders, all loadable via the encoder_factory() function.
"""

def attr_encoder_factory(model_name: str, **kwargs):
    """
    Instantiate a patch encoder model by name.

    This factory function dynamically maps model names to their corresponding encoder classes
    using a dictionary and generates a default `xxxAttrEncoder` class if not found.

    Args:
        model_name (str): Name of the encoder to instantiate.
        **kwargs: Optional keyword arguments passed directly to the encoder constructor.

    Returns:
        torch.nn.Module: An instance of the specified encoder model.

    Raises:
        ValueError: If `model_name` is not valid and cannot generate a class.
    """
    # Mapping of model names to class names
    model_mapping = {
        "uni_v1": "UNI",
        "gigapath": "GigaPath",
        "virchow": "Virchow",
        "conch_v1": "Conchv1",
        "conch_v15": "Conchv15"
    }

    # Check if the model name is valid
    class_name = model_mapping.get(model_name, None)
    if class_name is None:
        raise ValueError(f"Class {model_name} not found in defination")
    
    encoder_class_name = f"{class_name}AttrInferenceEncoder"
    # 有单独定义则直接使用单独定义，conch 需要单独提取视觉部分因此需要单独定义
    encoder_class = globals().get(encoder_class_name, None)
    if encoder_class is None:
        # 没有单独模型直接用生成器批量从 InferenceEncoder 生成
        encoder_class = make_encoder_class(class_name)
        base_model = globals().get(f"{class_name}InferenceEncoder")
        return encoder_class(**{"model": base_model})

    # Instantiate and return the encoder
    return encoder_class(**kwargs)


class Conchv1AttrInferenceEncoder(Conchv1InferenceEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_attr(self, x):
        model_viz = self.model.visual.trunk
        res = model_viz.forward_intermediates(x)
        attr = res[0][:, 1:, :]
        B, HW, C = attr.shape
        H, W = int(HW**0.5), int(HW**0.5)
        return attr.reshape(B, H, W, C)
    
    def forward(self, x):
        return self.forward_attr(x)

class Conchv15AttrInferenceEncoder(Conchv15InferenceEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_attr(self, x):
        model_viz = self.model.trunk
        res = model_viz.forward_features(x)
        attr = res[:, 1:, :]
        B, HW, C = attr.shape
        H, W = int(HW**0.5), int(HW**0.5)
        return attr.reshape(B, H, W, C)
    
    def forward(self, x):
        return self.forward_attr(x)


def make_encoder_class(name: str):
    """
    动态生成一个 subclass of {name}InferenceEncoder，
    名称为 {name}AttrInferenceEncoder，并注册到 globals()。
    """
    base_name = f"{name}InferenceEncoder"
    # 1) 从 globals() 拿到已有的基类
    try:
        BaseEncoder = globals()[base_name]
    except KeyError:
        raise ValueError(f"Cannot find base class {base_name} in globals()")

    new_name = f"{name}AttrInferenceEncoder"

    # 2) 定义新的 __init__：只接收 model，其它 kwargs 交给父类
    def __init__(self, model, *args, **kwargs):
        # 保存 model 到实例
        self.model = model
        # 只把 *args/**kwargs 传给父类，不包括 model
        super(type(self), self).__init__(*args, **kwargs)

    # 3) 定义 forward_attr，按照你给的逻辑来
    def forward_attr(self, x):
        res = self.model.forward_intermediates(x)
        attr = res[0][:, 1:, :]
        B, HW, C = attr.shape
        H = W = int(HW ** 0.5)
        return attr.reshape(B, H, W, C)

    def forward(self, x):
        return self.forward_attr(x)

    # 4) 创建新类并注册
    NewEncoder = type(
        new_name,
        (BaseEncoder,),
        {
            "__init__": __init__,
            "forward_attr": forward_attr,
            "forward": forward,
        },
    )
    globals()[new_name] = NewEncoder
    return NewEncoder


def expand_features_coords(features: np.ndarray, coords: np.ndarray):
    """
    Expands features and coordinates to pixel-level vectors and absolute coordinates.

    Args:
        features (np.ndarray): The original feature blocks with shape (N, H, W, C),
            where N is the number of blocks, H and W are the height and width of each block,
            and C is the number of channels.
        coords (np.ndarray): The top-left (row, col) coordinates of each block with shape (N, 2).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - expanded_features (np.ndarray): The expanded features array with shape (N * H * W, C).
            - expanded_coords (np.ndarray): The expanded coordinates array with shape (N * H * W, 2).

    Raises:
        ValueError: If the input shapes are invalid or incompatible.
    """
    N, H, W, C = features.shape
    expanded_features = features.reshape(N * H * W, C) #  1948 × 768,  1527232(1948 x 14 x 14) × 1024
    tile_size = H
    if coords.shape[0] > 1:
        tile_size = coords[1, 1] - coords[0, 1]  # 512

    offsets_row = np.repeat(np.linspace(0, tile_size, H+1)[0:H], W)
    offsets_col = np.tile(np.linspace(0, tile_size, W+1)[0:W], H)

    expanded_rows = np.repeat(coords[:, 0], H * W) + np.tile(offsets_col, N)
    expanded_cols = np.repeat(coords[:, 1], H * W) + np.tile(offsets_row, N)
    expanded_coords = np.stack([expanded_rows, expanded_cols], axis=1)

    return expanded_features, expanded_coords

def reshape_hdf5(
    in_path: str,
    out_path: str = None,
    features_key: str = 'features',
    coords_key: str = 'coords',
    new_features_key: str = 'features_attr',
    new_coords_key: str = 'coords_attr',
):
    """
    Reads an HDF5 file, expands the (N, H, W, C) features and (N, 2) coordinates,
    and writes the expanded data back to the HDF5 file.

    Args:
        in_path (str): Path to the input HDF5 file. Must contain datasets specified
            by `features_key` and `coords_key`.
        out_path (str, optional): Path to the output HDF5 file. If None, the expanded
            data is appended to the input file. Defaults to None.
        features_key (str): Key for the original features dataset in the HDF5 file.
            Defaults to 'features'.
        coords_key (str): Key for the original coordinates dataset in the HDF5 file.
            Defaults to 'coords'.
        new_features_key (str): Key for the expanded features dataset to be written
            to the HDF5 file. Defaults to 'features_'.
        new_coords_key (str): Key for the expanded coordinates dataset to be written
            to the HDF5 file. Defaults to 'coords_att'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - expanded_features (np.ndarray): The expanded features array with shape (N * H * W, C).
            - expanded_coords (np.ndarray): The expanded coordinates array with shape (N * H * W, 2).

    """
    f_in = h5py.File(in_path, 'r')
    feats = f_in[features_key][:]
    coords = f_in[coords_key][:]
    exp_feats, exp_coords = expand_features_coords(feats, coords)
    if out_path is None:
        f_out = f_in
    else:
        f_in.close()
        f_out = h5py.File(out_path, 'w')

    if new_features_key in f_out:
        del f_out[new_features_key]
    f_out.create_dataset(new_features_key, data=exp_feats, dtype=feats.dtype)
    if new_coords_key in f_out:
        del f_out[new_coords_key]
    f_out.create_dataset(new_coords_key, data=exp_coords, dtype=coords.dtype)

    f_out.close()
    if out_path:
        f_in.close()
    return exp_feats, exp_coords
