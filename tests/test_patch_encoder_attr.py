import os
import unittest

import torch
import numpy as np 
from PIL import Image
import h5py

import sys; sys.path.append('../')
from trident.patch_encoder_models import *
from trident.patch_encoder_models.loader_attr import attr_encoder_factory, reshape_hdf5

"""
Test forward pass of patch encoders
"""

class TestPatchEncoders(object):

    def setUp(self):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device  = torch.device("cpu")
        self.dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        self.dummy_image = Image.fromarray(self.dummy_image)

    def _test_encoder_attr_forward(self, encoder_name, **kwargs):
        print("\033[95m" + f"Testing {encoder_name} forward pass" + "\033[0m")
        if kwargs:
            print("\033[92m" + f"    With kwargs: {kwargs}" + "\033[0m")
        encoder = attr_encoder_factory(encoder_name, **kwargs)
        encoder = encoder.to(self.device)
        encoder.eval()

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=encoder.precision):
            dummy_input = encoder.eval_transforms(self.dummy_image).to(self.device).unsqueeze(dim=0)
            output = encoder.forward_attr(dummy_input)

        # self.assertIsNotNone(output)
        # self.assertIsInstance(output, torch.Tensor)
        print("\033[94m"+ f"    {encoder_name} forward pass success with output {output.shape}" + "\033[0m")
    
    def test_conch_v1_attr_forward(self):
        self._test_encoder_attr_forward('conch_v1', with_proj = True, normalize = True)
        self._test_encoder_attr_forward('conch_v1', with_proj = False, normalize = True)
        self._test_encoder_attr_forward('conch_v1', with_proj = True, normalize = False)
        self._test_encoder_attr_forward('conch_v1', with_proj = False, normalize = False)
        
    def test_other_attr_forward(self):
        self._test_encoder_attr_forward('conch_v15')
        # self._test_encoder_attr_forward('uni_v1')
        # self._test_encoder_attr_forward('virchow')
        # self._test_encoder_attr_forward('gigapath')
    

    def test_expand_and_reshape_hdf5(self):
        # 临时文件
        tmp_path = "./"
        in_file = tmp_path / "test.h5"
        N, H, W, C = 3, 2, 2, 5
        feats = np.arange(N * H * W * C, dtype=np.float32).reshape(N, H, W, C)
        coords = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int64)

        with h5py.File(in_file, 'w') as f:
            f.create_dataset('features', data=feats)
            f.create_dataset('coords', data=coords)

        out_file = tmp_path / "out.h5"
        exp_feats, exp_coords = reshape_hdf5(str(in_file), str(out_file))

        assert exp_feats.shape == (N * H * W, C)
        assert exp_coords.shape == (N * H * W, 2)

        np.testing.assert_array_equal(exp_feats[: H * W], feats[0].reshape(-1, C))
        expected_offsets = np.array([[0,0],[0,1],[1,0],[1,1]])
        np.testing.assert_array_equal(exp_coords[: H * W], coords[0] + expected_offsets)

        np.testing.assert_array_equal(exp_coords[- H * W :], coords[-1] + expected_offsets)
        os.remove(in_file)
        os.remove(out_file)

if __name__ == '__main__':
    # unittest.main()
    obj = TestPatchEncoders()
    obj.setUp()
    # obj.test_other_attr_forward()
    obj.test_expand_and_reshape_hdf5
    # obj.test_conch_v1_attr_forward()
    