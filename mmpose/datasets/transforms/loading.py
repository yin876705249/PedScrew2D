# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import mmcv
from mmcv.transforms import LoadImageFromFile
from scipy.ndimage import distance_transform_edt

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        try:
            if 'img' not in results:
                # Load image from file by :meth:`LoadImageFromFile.transform`
                results = super().transform(results)
            else:
                img = results['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results:
                    results['img_path'] = None
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]
        except Exception as e:
            e = type(e)(
                f'`{str(e)}` occurs when loading `{results["img_path"]}`.'
                'Please check whether the file exists.')
            raise e

        return results


@TRANSFORMS.register_module()
class LoadImageAndMask(LoadImage):
    """Load an image and its segmentation mask from file, optionally applying distance transform to the mask."""

    def __init__(self, 
                 mask_to_float32=False, 
                 mask_color_type='unchanged', 
                 apply_distance_transform=False, 
                 distance_threshold=5, 
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_to_float32 = mask_to_float32
        self.mask_color_type = mask_color_type
        self.apply_distance_transform = apply_distance_transform
        self.distance_threshold = distance_threshold

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function that loads both image and mask, and optionally applies distance transform to mask."""
        # Load image using the superclass method
        results = super().transform(results)

        # Load mask from file
        try:
            if 'gt_sem_seg' not in results:  # 使用 gt_sem_seg 作为键
                mask = mmcv.imread(results['mask_path'], flag=self.mask_color_type)
                if self.mask_to_float32:
                    mask = mask.astype(np.float32)
                if self.apply_distance_transform:
                    mask = self.apply_distance_transform_to_mask(mask)
                results['gt_sem_seg'] = mask  # 将掩码存储为 gt_sem_seg
                results['mask_shape'] = mask.shape
            else:
                mask = results['gt_sem_seg']  # 直接使用 gt_sem_seg
                assert isinstance(mask, np.ndarray)
                if self.apply_distance_transform:
                    mask = self.apply_distance_transform_to_mask(mask)
                results['gt_sem_seg'] = mask  # 更新 gt_sem_seg
        except Exception as e:
            e = type(e)(
                f'{str(e)} occurs when loading mask `{results["mask_path"]}`.'
                'Please check whether the mask file exists.')
            raise e

        return results

    def apply_distance_transform_to_mask(self, mask):
        """Apply distance transform to each class in the mask."""
        transformed_mask = np.zeros_like(mask)
        classes = np.unique(mask)  # Get all unique classes from the mask
        for cls in classes:
            if cls == 0:  # Typically, class 0 is the background.
                continue
            class_mask = (mask == cls).astype(np.uint8)
            # Compute the distance transform
            dist_transform = distance_transform_edt(class_mask)
            # Threshold the distance transform to find edges close to the boundary
            edge_mask = (dist_transform < self.distance_threshold)
            transformed_mask[edge_mask] = 1
        return transformed_mask