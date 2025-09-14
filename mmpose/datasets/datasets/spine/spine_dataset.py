# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional

import numpy as np

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


# @DATASETS.register_module()
# class SpineDataset(BaseCocoStyleDataset):
#     """Custom dataset for spine keypoint detection without bbox info."""

#     METAINFO: dict = dict(
#         from_file='configs/_base_/datasets/spine.py')

#     def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
#         """Parse raw annotation of an instance from the custom dataset.

#         Args:
#             raw_data_info (dict): Raw data information loaded from
#                 ``ann_file``. It should have following contents:

#                 - ``'raw_ann_info'``: Raw annotation of an instance
#                 - ``'raw_img_info'``: Raw information of the image that
#                     contains the instance

#         Returns:
#             dict: Parsed instance annotation
#         """

#         ann = raw_data_info['raw_ann_info']
#         img = raw_data_info['raw_img_info']

#         # Construct the image path
#         img_path = osp.join(self.data_prefix['img'], img['file_name'])
#         img_w, img_h = img['width'], img['height']

#         # Since there is no bounding box info in the custom dataset,
#         # we use the whole image as the bounding box.
#         # The bbox should be in the format [x, y, width, height]
#         # and it should be wrapped in a list to match expected input shape.
#         bbox = np.array([5, 5, img_w - 5, img_h - 5], dtype=np.float32).reshape(1, 4)

#         # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
#         _keypoints = np.array(
#             ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
#         keypoints = _keypoints[..., :2]
#         keypoints_visible = np.minimum(1, _keypoints[..., 2])

#         if 'num_keypoints' in ann:
#             num_keypoints = ann['num_keypoints']
#         else:
#             num_keypoints = np.count_nonzero(keypoints.max(axis=2))

#         data_info = {
#             'img_id': img['id'],
#             'img_path': img_path,
#             'bbox': bbox,
#             'bbox_score': np.ones(1, dtype=np.float32),
#             'num_keypoints': num_keypoints,
#             'keypoints': keypoints,
#             'keypoints_visible': keypoints_visible,
#             'id': ann['id'],
#             'category_id': ann['category_id']
#         }
#         return data_info
    
    
    
@DATASETS.register_module()
class SpineDataset(BaseCocoStyleDataset):
    """Custom dataset for spine keypoint detection with mask support."""

    METAINFO: dict = dict(
        from_file='configs/_base_/datasets/spine.py')

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw annotation of an instance from the custom dataset.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict: Parsed instance annotation including additional mask info.
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # Construct the image path and corresponding mask path
        img_path = osp.join(self.data_prefix['img'], img['file_name'])
        
        mask_path = None
        if 'seg' in self.data_prefix:
            mask_file_name = img['file_name'].replace('.jpg', '_mask.png')
            mask_path = osp.join(self.data_prefix['seg'], mask_file_name)

        img_w, img_h = img['width'], img['height']

        # Use the whole image as the bounding box
        bbox = np.array([5, 5, img_w - 10, img_h - 10], dtype=np.float32).reshape(1, 4)

        # Process keypoints
        _keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])
        num_keypoints = ann.get('num_keypoints', np.count_nonzero(keypoints.max(axis=2)))

        data_info = {
            'img_id': img['id'],
            'img_path': img_path,
            'mask_path': mask_path,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'id': ann['id'],
            'category_id': ann['category_id']
        }

        return data_info