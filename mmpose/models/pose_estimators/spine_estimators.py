import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List


from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from .topdown import TopdownPoseEstimator
from mmseg.models import FCNHead
from mmpose.registry import MODELS

# @MODELS.register_module()
# class PoseWithSegmentation(TopdownPoseEstimator):
#     def __init__(self, seg_head=None, joint_edge_loss=None, **kwargs):
#         super().__init__(**kwargs)
#         if seg_head is not None:
#             self.seg_head = FCNHead(**seg_head)
#         if joint_edge_loss is not None:
#             self.joint_edge_loss = MODELS.build(joint_edge_loss)

#     def forward(self, inputs, data_samples=None, mode='tensor'):
#         """Unified interface for forward pass.

#         Args:
#             inputs (torch.Tensor): The input data as tensors.
#             data_samples (list, optional): The data samples.
#             mode (str, optional): Mode of operation, can be 'tensor', 'loss', or 'predict'.

#         Returns:
#             Depending on the mode:
#             - 'tensor': Returns the raw output from the network.
#             - 'predict': Returns the post-processed predictions.
#             - 'loss': Returns the computed losses.
#         """
#         # 调用基类的forward方法处理主要的pose estimation任务
#         pose_outputs = super().forward(inputs, data_samples, mode)

#         # 如果定义了分割头，处理分割任务
#         if hasattr(self, 'seg_head'):
#             if hasattr(self, 'neck'):
#                 fpn_features = self.neck(self.backbone(inputs))
#                 seg_outputs = self.seg_head(fpn_features[3])
#             else:
#                 seg_outputs = self.seg_head(self.backbone(inputs))

#             if mode == 'tensor':
#                 return pose_outputs, seg_outputs
#             elif mode == 'loss':
#                 # 从data_samples中提取分割掩码和关键点
#                 seg_masks = torch.stack([sample.gt_instances.mask for sample in data_samples])  # 提取分割掩码
#                 keypoints = torch.stack([torch.tensor(sample.gt_instances.keypoints) for sample in data_samples])  # 提取关键点

#                 # 计算分割任务的损失
#                 # seg_loss = self.seg_head.loss(seg_outputs, seg_masks)
                
#                 # 计算联合损失
                
#                 print(pose_outputs.keys())
#                 if hasattr(self, 'joint_edge_loss'):
#                     joint_loss = self.joint_edge_loss(
#                         pose_outputs['pred_keypoints'],  # 预测的关键点
#                         keypoints,                       # 真实的关键点
#                         seg_outputs,                     # 预测的分割掩码
#                         seg_masks                        # 真实的分割掩码
#                     )
#                     total_loss = pose_outputs['loss'] + seg_outputs['loss'] + joint_loss
#                 else:
#                     total_loss = pose_outputs['loss'] + seg_outputs['loss']
                
#                 return total_loss

#         return pose_outputs


@MODELS.register_module()
class PoseWithSegmentation(TopdownPoseEstimator):
    """Pose Estimator with additional segmentation head."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 seg_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            metainfo=metainfo)
        if seg_head is not None:
            self.seg_head = MODELS.build(seg_head)

    @property
    def with_seg_head(self) -> bool:
        """bool: whether the pose estimator has a segmentation head."""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples, including segmentation loss."""
        feats = self.extract_feat(inputs)

        losses = dict()

        if self.with_head:
            keypoint_losses = self.head.loss(feats, data_samples, train_cfg=self.train_cfg)
            losses.update(keypoint_losses)
        
        if self.with_seg_head:
            seg_losses = self.seg_head.loss(feats, data_samples, train_cfg=self.train_cfg)
            losses.update(seg_losses)
        return losses

    # def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
    #     """Predict results from a batch of inputs and data samples with post-processing, including segmentation."""
    #     if isinstance(inputs, tuple):
    #         # Assume that we want the first element of the tuple for subsequent processing
    #         inputs = inputs[0]
    #     print(inputs.size())
    #     feats = self.extract_feat(inputs)
    #     # print(len(feats))
    #     # print(len(data_samples))
    #     # print(feats[0].size())
    #     # print(data_samples[0].gt_sem_seg.size())

    #     pose_results = super().predict(feats, data_samples)

    #     if self.with_seg_head:
    #         seg_pred = self.seg_head.get_seg_masks(feats, data_samples)
    #         for pose_result, seg_mask in zip(pose_results, seg_pred):
    #             pose_result.pred_masks = seg_mask

    #     return pose_results

