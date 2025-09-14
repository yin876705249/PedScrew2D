from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.functional as TF

from mmpose.registry import MODELS


@MODELS.register_module()
class SpineLoss(nn.Module):
    """Combined loss for heatmap and geometric consistency.

    Args:
        heatmap_loss (dict): Config for heatmap loss.
        geo_loss (dict): Config for geometric consistency loss.
    """

    def __init__(self, heatmap_loss: dict, line_loss: dict):
        super().__init__()
        self.heatmap_loss = MODELS.build(heatmap_loss)
        self.line_loss = MODELS.build(line_loss)

    def forward(self, output: Tensor, target: Tensor, target_weights: Optional[Tensor] = None) -> Tensor:
        """Forward function of loss.

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W].
            target (Tensor): The target heatmaps with shape [B, K, H, W].
            target_weights (Tensor, optional): The target weights of different
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).

        Returns:
            Tensor: The calculated loss.
        """
        # Calculate heatmap loss
        heatmap_loss = self.heatmap_loss(output, target, target_weights)
        # Calculate geometric consistency loss
        line_loss = self.line_loss(output, target, target_weights)

        # Combine losses
        total_loss = heatmap_loss + line_loss
        return total_loss


@MODELS.register_module()
class GeometricConsistencyLoss(nn.Module):
    """
    几何一致性损失（基于 MND、长度误差和角度误差），用于关键点检测。

    该损失强制模型在关键点之间的相对关系上保持预测关键点与真实关键点的一致性，
    通过计算关键点段之间的修改后的归一化距离（MND）、长度误差和角度误差。

    Args:
        skeleton_info (List[Tuple[int, int]]): 定义骨架连接的关键点索引对。
        k (float): MND 的缩放因子，控制负指数函数的形状。默认为 1.0。
        loss_weight (float): 几何一致性损失的整体权重。默认为 1.0。
        mnd_weight (float): MND 损失的权重。默认为 1.0。
        length_weight (float): 长度误差损失的权重。默认为 1.0。
        angle_weight (float): 角度误差损失的权重。默认为 1.0。
    """

    def __init__(self,
                 skeleton_info: List[Tuple[int, int]],
                 k: float = 1.0,
                 loss_weight: float = 1.0,
                 AS_weight: float = 1.0,
                 LS_weight: float = 1.0,
                 DS_weight: float = 1.0
                 
                 ):
        super(GeometricConsistencyLoss, self).__init__()
        self.skeleton_info = skeleton_info
        self.k = k
        self.loss_weight = loss_weight
        self.DS_weight = DS_weight
        self.LS_weight = LS_weight
        self.AS_weight = AS_weight

    def point_to_segment_distance(self, point: torch.Tensor, 
                                  seg_start: torch.Tensor, 
                                  seg_end: torch.Tensor) -> torch.Tensor:
        """
        计算点到线段的最短距离。

        Args:
            point (Tensor): 形状为 [N, 2] 的点坐标。
            seg_start (Tensor): 形状为 [N, 2] 的线段起点坐标。
            seg_end (Tensor): 形状为 [N, 2] 的线段终点坐标。

        Returns:
            Tensor: 点到线段的距离，形状为 [N]。
        """
        seg_vector = seg_end - seg_start  # [N, 2]
        point_vector = point - seg_start  # [N, 2]
        seg_len_sq = torch.sum(seg_vector ** 2, dim=-1)  # [N]
        seg_len_sq = torch.clamp(seg_len_sq, min=1e-8)
        proj_length = torch.sum(point_vector * seg_vector, dim=-1) / seg_len_sq  # [N]
        proj_length_clamped = torch.clamp(proj_length, 0.0, 1.0).unsqueeze(-1)  # [N, 1]
        nearest_point = seg_start + proj_length_clamped * seg_vector  # [N, 2]
        distance = torch.norm(point - nearest_point, dim=-1)  # [N]
        return distance

    def compute_mnd(self, pred_segment: torch.Tensor, gt_segment: torch.Tensor) -> torch.Tensor:
        """
        计算预测段与真实段之间的修改后的归一化距离（MND）。

        Args:
            pred_segment (Tensor): 预测的段关键点坐标，形状为 [N, 2, 2]。
            gt_segment (Tensor): 真实的段关键点坐标，形状为 [N, 2, 2]。

        Returns:
            Tensor: MND 损失，形状为 [N]。
        """
        # 计算预测段的两个端点到真实段的距离
        dist_pred_to_gt_start = self.point_to_segment_distance(pred_segment[:, 0, :], 
                                                                gt_segment[:, 0, :], 
                                                                gt_segment[:, 1, :])
        dist_pred_to_gt_end = self.point_to_segment_distance(pred_segment[:, 1, :], 
                                                              gt_segment[:, 0, :], 
                                                              gt_segment[:, 1, :])
        # 计算真实段的两个端点到预测段的距离
        dist_gt_to_pred_start = self.point_to_segment_distance(gt_segment[:, 0, :], 
                                                                pred_segment[:, 0, :], 
                                                                pred_segment[:, 1, :])
        dist_gt_to_pred_end = self.point_to_segment_distance(gt_segment[:, 1, :], 
                                                              pred_segment[:, 0, :], 
                                                              pred_segment[:, 1, :])
        # 拼接所有距离
        distances = torch.stack([dist_pred_to_gt_start, dist_pred_to_gt_end,
                                 dist_gt_to_pred_start, dist_gt_to_pred_end], dim=-1)  # [N, 4]

        # 计算真实段的长度
        gt_lengths = torch.norm(gt_segment[:, 1, :] - gt_segment[:, 0, :], dim=-1)  # [N]
        gt_lengths = torch.clamp(gt_lengths, min=1e-8)  # 避免除以零

        # 归一化距离
        normalized_distances = distances / gt_lengths.unsqueeze(-1)  # [N, 4]

        # 应用负指数函数
        exp_scores = torch.exp(-normalized_distances / self.k)  # [N, 4]

        # 计算每个段的平均得分
        mnd_score = exp_scores.mean(dim=-1)  # [N]

        # 将得分转换为损失，得分越高（距离越小），损失越低
        loss = 1.0 - mnd_score  # [N]

        return loss  # [N]

    def compute_length_error(self, pred_segment: torch.Tensor, gt_segment: torch.Tensor) -> torch.Tensor:
        """
        计算预测段与真实段长度之间的误差。

        Args:
            pred_segment (Tensor): 预测的段关键点坐标，形状为 [N, 2, 2]。
            gt_segment (Tensor): 真实的段关键点坐标，形状为 [N, 2, 2]。

        Returns:
            Tensor: 长度误差损失，形状为 [N]。
        """
        pred_length = torch.norm(pred_segment[:, 1, :] - pred_segment[:, 0, :], dim=-1)  # [N]
        gt_length = torch.norm(gt_segment[:, 1, :] - gt_segment[:, 0, :], dim=-1)      # [N]
        gt_length = torch.clamp(gt_length, min=1e-8)  # 避免除以零

        length_error = torch.abs(pred_length - gt_length) / gt_length  # [N]
        # 应用负指数函数以获得损失
        length_error_score = torch.exp(-length_error / self.k)  # [N]
        # 损失为 1 - score，得分越高（误差越小），损失越低
        loss = 1.0 - length_error_score  # [N]
        return loss  # [N]

    def compute_angle_error(self, pred_segment: torch.Tensor, gt_segment: torch.Tensor) -> torch.Tensor:
        """
        计算预测段与真实段之间的角度误差。

        Args:
            pred_segment (Tensor): 预测的段关键点坐标，形状为 [N, 2, 2]。
            gt_segment (Tensor): 真实的段关键点坐标，形状为 [N, 2, 2]。

        Returns:
            Tensor: 角度误差损失，形状为 [N]。
        """
        pred_vector = pred_segment[:, 1, :] - pred_segment[:, 0, :]  # [N, 2]
        gt_vector = gt_segment[:, 1, :] - gt_segment[:, 0, :]      # [N, 2]

        pred_vector_norm = F.normalize(pred_vector, p=2, dim=1)    # [N, 2]
        gt_vector_norm = F.normalize(gt_vector, p=2, dim=1)        # [N, 2]

        dot_product = torch.sum(pred_vector_norm * gt_vector_norm, dim=1).clamp(-1.0, 1.0)  # [N]
        angle_error_rad = torch.acos(dot_product)  # [N]
        angle_error_deg = angle_error_rad * (180.0 / torch.pi)  # [N]

        # 将角度误差转换为损失，误差越小，损失越低
        # 可以使用 MSE 或其他适合的函数
        loss = angle_error_deg  # [N]
        return loss  # [N]

    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                target_weights: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            output (Tensor): 预测的热图，形状为 [B, K, H, W]。
            target (Tensor): 真实的热图，形状为 [B, K, H, W]。
            target_weights (Tensor, optional): 每个关键点的权重，形状为 [B, K] 或 [B, K, 1]。默认为 None。
            mask (Tensor, optional): 有效段的掩码，形状为 [B, S]，其中 S 是段的数量。默认为 None。

        Returns:
            Tensor: 计算得到的几何一致性损失。
        """
        # 将热图转换为关键点坐标
        pred_coords = self.heatmaps_to_coords(output)  # [B, K, 2]
        gt_coords = self.heatmaps_to_coords(target)     # [B, K, 2]

        B, K, _ = pred_coords.shape
        S = len(self.skeleton_info)

        # 应用目标权重
        if target_weights is not None:
            # 假设 target_weights 的形状为 [B, K] 或 [B, K, 1]
            if target_weights.dim() == 3:
                weight = target_weights.squeeze(-1)  # [B, K]
            else:
                weight = target_weights  # [B, K]
        else:
            # 如果没有提供权重，则所有权重为 1
            weight = torch.ones_like(target[:, :, 0])  # [B, K]

        # 初始化段掩码，形状为 [B, S]
        segment_mask = torch.ones((B, S), device=output.device, dtype=output.dtype)  # [B, S]

        for s, (start_idx, end_idx) in enumerate(self.skeleton_info):
            # 更新段掩码，只有起点和终点权重都非零时，段才有效
            segment_mask[:, s] = weight[:, start_idx] * weight[:, end_idx]  # [B]

        # 如果提供了额外的掩码，则应用它
        if mask is not None:
            # 假设 mask 的形状为 [B, S]
            segment_mask = segment_mask * mask  # [B, S]

        # 提取预测和真实段的起点和终点坐标
        pred_segments_start = pred_coords[:, [pair[0] for pair in self.skeleton_info], :]  # [B, S, 2]
        pred_segments_end = pred_coords[:, [pair[1] for pair in self.skeleton_info], :]    # [B, S, 2]
        gt_segments_start = gt_coords[:, [pair[0] for pair in self.skeleton_info], :]      # [B, S, 2]
        gt_segments_end = gt_coords[:, [pair[1] for pair in self.skeleton_info], :]        # [B, S, 2]

        # 组合起点和终点坐标，形状为 [B, S, 2, 2]
        pred_segments = torch.stack([pred_segments_start, pred_segments_end], dim=2)  # [B, S, 2, 2]
        gt_segments = torch.stack([gt_segments_start, gt_segments_end], dim=2)        # [B, S, 2, 2]

        # 重塑为 [B*S, 2, 2] 以便批量计算
        pred_segments = pred_segments.view(-1, 2, 2)  # [B*S, 2, 2]
        gt_segments = gt_segments.view(-1, 2, 2)      # [B*S, 2, 2]
        segment_mask = segment_mask.view(-1)          # [B*S]

        # 仅保留有效的段
        valid_indices = segment_mask > 0  # [B*S]
        if valid_indices.sum() == 0:
            # 如果没有有效的段，返回零损失
            return torch.tensor(0.0, device=output.device, requires_grad=True)

        pred_segments = pred_segments[valid_indices]  # [N, 2, 2]
        gt_segments = gt_segments[valid_indices]      # [N, 2, 2]

        if pred_segments.numel() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)

        # 计算 MND 损失
        mnd_loss = self.compute_mnd(pred_segments, gt_segments)  # [N]

        # 计算长度误差损失
        length_loss = self.compute_length_error(pred_segments, gt_segments)  # [N]

        # 计算角度误差损失
        angle_loss = self.compute_angle_error(pred_segments, gt_segments)    # [N]

        # 归一化各项损失
        mnd_loss_mean = mnd_loss.mean()
        length_loss_mean = length_loss.mean()
        angle_loss_mean = angle_loss.mean()

        # 组合所有损失，使用各自的权重
        total_loss = (self.DS_weight * mnd_loss_mean +
                      self.LS_weight * length_loss_mean +
                      self.AS_weight * angle_loss_mean)

        return total_loss * self.loss_weight

    def heatmaps_to_coords(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        使用 Soft-ArgMax 将热图转换为关键点坐标。

        Args:
            heatmaps (Tensor): 热图，形状为 [B, K, H, W]。

        Returns:
            Tensor: 关键点坐标，形状为 [B, K, 2]，格式为 [x, y]。
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device

        # 创建坐标网格
        x = torch.linspace(0, W - 1, W, device=device)
        y = torch.linspace(0, H - 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # 对于 PyTorch >= 1.10
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # 对热图进行 Softmax 归一化
        heatmaps = heatmaps.view(B, K, -1)
        heatmaps = F.softmax(heatmaps, dim=-1)
        heatmaps = heatmaps.view(B, K, H, W)

        # 计算期望坐标
        expected_x = torch.sum(heatmaps * grid_x, dim=[2, 3])  # [B, K]
        expected_y = torch.sum(heatmaps * grid_y, dim=[2, 3])  # [B, K]

        coords = torch.stack([expected_x, expected_y], dim=2)  # [B, K, 2]

        return coords
    
# @MODELS.register_module()
# class GeometricConsistencyLoss(nn.Module):
#     """
#     几何一致性损失，用于关键点检测。

#     该损失强制模型在距离、方向和相对位置上保持预测关键点与真实关键点的一致性。

#     Args:
#         skeleton_info (list of tuples): 定义骨架连接的关键点索引对。
#         loss_weight (float): 几何一致性损失的权重。默认为 1.0。
#     """

#     def __init__(self, skeleton_info, loss_weight=1.0):
#         super().__init__()
#         self.skeleton_info = skeleton_info
#         self.loss_weight = loss_weight

#     def forward(self, output: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
#         """
#         前向传播函数。

#         Args:
#             output (Tensor): 预测的热图，形状为 [B, K, H, W]。
#             target (Tensor): 真实的热图，形状为 [B, K, H, W]。
#             *args: 其他位置参数（未使用）。
#             **kwargs: 关键点权重和掩码。
#                 - target_weights (Tensor, optional): 每个关键点的权重，形状为 [B, K]。
#                 - mask (Tensor, optional): 有效热图像素的掩码，形状为 [B, K, H, W]。

#         Returns:
#             Tensor: 计算得到的几何一致性损失。
#         """
#         target_weights = kwargs.get('target_weights', None)
#         mask = kwargs.get('mask', None)

#         # 将热图转换为关键点坐标
#         pred_coords = self.heatmaps_to_coords(output)
#         target_coords = self.heatmaps_to_coords(target)

#         # 应用目标权重
#         if target_weights is not None:
#             mask_weight = target_weights.unsqueeze(-1)
#             pred_coords = pred_coords * mask_weight.unsqueeze(-1)
#             target_coords = target_coords * mask_weight.unsqueeze(-1)

#         # 初始化总损失
#         loss = 0.0

#         for pair in self.skeleton_info:
#             idx_start, idx_end = pair

#             # 预测的起点和终点坐标
#             pred_start = pred_coords[:, idx_start, :]  # [B, 2]
#             pred_end = pred_coords[:, idx_end, :]      # [B, 2]

#             # 真实的起点和终点坐标
#             target_start = target_coords[:, idx_start, :]  # [B, 2]
#             target_end = target_coords[:, idx_end, :]      # [B, 2]

#             # 1. 距离约束
#             # pred_distance = torch.norm(pred_end - pred_start, dim=1)  # [B]
#             # target_distance = torch.norm(target_end - target_start, dim=1)  # [B]
#             # distance_loss = F.l1_loss(pred_distance, target_distance, reduction='mean')

#             # # 2. 方向一致性
#             # pred_dir = pred_end - pred_start  # [B, 2]
#             # target_dir = target_end - target_start  # [B, 2]

#             # # 归一化方向向量
#             # pred_dir_norm = F.normalize(pred_dir, p=2, dim=1)  # [B, 2]
#             # target_dir_norm = F.normalize(target_dir, p=2, dim=1)  # [B, 2]
#             # direction_loss = F.mse_loss(pred_dir_norm, target_dir_norm, reduction='mean')

#             # 3. 相对位置约束
#             # 相对位置定义为预测相对于真实相对位置的差异
#             # 即 (pred_end - pred_start) 应尽量接近 (target_end - target_start)
#             relative_pos_pred = pred_end - pred_start  # [B, 2]
#             relative_pos_target = target_end - target_start  # [B, 2]
#             relative_position_loss = F.mse_loss(relative_pos_pred, relative_pos_target, reduction='mean')

#             # 合并各项损失
#             pair_loss = relative_position_loss
#             loss += pair_loss

#         # 平均所有关键点对的损失
#         loss = loss / len(self.skeleton_info)
#         return loss * self.loss_weight

#     def heatmaps_to_coords(self, heatmaps: Tensor) -> Tensor:
#         """
#         使用 Soft-ArgMax 将热图转换为关键点坐标。

#         Args:
#             heatmaps (Tensor): 热图，形状为 [B, K, H, W]。

#         Returns:
#             Tensor: 关键点坐标，形状为 [B, K, 2]，格式为 [x, y]。
#         """
#         B, K, H, W = heatmaps.shape
#         device = heatmaps.device

#         # 创建坐标网格
#         x = torch.linspace(0, W - 1, W, device=device)
#         y = torch.linspace(0, H - 1, H, device=device)
#         grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # 对于 PyTorch >= 1.10
#         grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
#         grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

#         # 对热图进行 Softmax 归一化
#         heatmaps = heatmaps.view(B, K, -1)
#         heatmaps = F.softmax(heatmaps, dim=-1)
#         heatmaps = heatmaps.view(B, K, H, W)

#         # 计算期望坐标
#         expected_x = torch.sum(heatmaps * grid_x, dim=[2, 3])  # [B, K]
#         expected_y = torch.sum(heatmaps * grid_y, dim=[2, 3])  # [B, K]

#         coords = torch.stack([expected_x, expected_y], dim=2)  # [B, K, 2]

#         return coords
    


@MODELS.register_module()
class LineMSELoss(nn.Module):
    """MSE loss for skeleton line segments heatmaps.

    This loss computes the Mean Squared Error between the predicted and target
    skeleton line segments heatmaps based on the keypoint heatmaps.

    Args:
        skeleton_info (List[Tuple[int, int]]): List of tuples defining which keypoints form a skeleton line.
        line_width (int): The width of the line segments in pixels. Defaults to 1.
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 skeleton_info: List[Tuple[int, int]],
                 line_width: int = 1,
                 loss_weight: float = 1.0):
        super().__init__()
        self.skeleton_info = skeleton_info
        self.line_width = line_width
        self.loss_weight = loss_weight

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """Forward function of loss.

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of different
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise). Defaults to ``None``
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        # Generate predicted line heatmaps from output keypoint heatmaps
        pred_line_heatmaps = self._generate_line_heatmaps(output)

        # Generate target line heatmaps from target keypoint heatmaps
        target_line_heatmaps = self._generate_line_heatmaps(target)

        # Apply mask if provided
        if mask is not None:
            # Assuming mask shape is [B, 1, H, W] or [B, K, H, W]
            if mask.size(1) == 1:
                pred_line_heatmaps = pred_line_heatmaps * mask
                target_line_heatmaps = target_line_heatmaps * mask
            else:
                # If mask has per-keypoint information
                pred_line_heatmaps = pred_line_heatmaps * mask
                target_line_heatmaps = target_line_heatmaps * mask

        # Compute MSE loss between predicted and target line heatmaps
        loss = F.mse_loss(pred_line_heatmaps, target_line_heatmaps, reduction='mean')

        return loss * self.loss_weight

    def _generate_line_heatmaps(self, keypoint_heatmaps: Tensor) -> Tensor:
        """Generate line segments heatmaps from keypoint heatmaps.

        Args:
            keypoint_heatmaps (Tensor): Keypoint heatmaps with shape [B, K, H, W]

        Returns:
            Tensor: Line segments heatmaps with shape [B, L, H, W], where L is the number of line segments.
        """
        B, K, H, W = keypoint_heatmaps.shape
        L = len(self.skeleton_info)
        device = keypoint_heatmaps.device
        dtype = keypoint_heatmaps.dtype

        # Extract keypoint coordinates using soft-argmax
        coords = self._extract_coordinates(keypoint_heatmaps)  # [B, K, 2]

        # Initialize line heatmaps
        line_heatmaps = torch.zeros((B, L, H, W), device=device, dtype=dtype)

        for i, (kp_start, kp_end) in enumerate(self.skeleton_info):
            # Get start and end coordinates
            start_coords = coords[:, kp_start, :]  # [B, 2]
            end_coords = coords[:, kp_end, :]      # [B, 2]

            # Generate line heatmap for each sample in the batch
            for b in range(B):
                line_heatmaps[b, i] = self._draw_line_heatmap(start_coords[b], end_coords[b], H, W)

        return line_heatmaps

    def _draw_line_heatmap(self, start: Tensor, end: Tensor, H: int, W: int) -> Tensor:
        """Draw a line between two points on a heatmap.

        Args:
            start (Tensor): Starting coordinate [x, y]
            end (Tensor): Ending coordinate [x, y]
            H (int): Height of the heatmap
            W (int): Width of the heatmap

        Returns:
            Tensor: Line heatmap with shape [H, W]
        """
        # Create a mesh grid
        yy, xx = torch.meshgrid(torch.arange(0, H, device=start.device),
                                torch.arange(0, W, device=start.device))
        xx = xx.float()
        yy = yy.float()

        # Compute distance to the line segment
        x0, y0 = start
        x1, y1 = end
        numerator = torch.abs((y1 - y0) * xx - (x1 - x0) * yy + x1 * y0 - y1 * x0)
        denominator = torch.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2) + 1e-6  # Prevent division by zero
        distance = numerator / denominator

        # Convert distance to heatmap using Gaussian
        sigma = self.line_width / 2
        line_heatmap = torch.exp(-(distance ** 2) / (2 * sigma ** 2))

        return line_heatmap

    def _extract_coordinates(self, heatmaps: Tensor) -> Tensor:
        """Extract keypoint coordinates from heatmaps using soft-argmax.

        Args:
            heatmaps (Tensor): Heatmaps with shape [B, K, H, W]

        Returns:
            Tensor: Coordinates with shape [B, K, 2] in (x, y) format
        """
        B, K, H, W = heatmaps.shape

        # Create coordinate grids
        device = heatmaps.device
        y = torch.linspace(0, H - 1, H, device=device).view(1, 1, H, 1).expand(B, K, H, W)
        x = torch.linspace(0, W - 1, W, device=device).view(1, 1, 1, W).expand(B, K, H, W)

        # Apply softmax to heatmaps
        heatmaps = F.softmax(heatmaps.view(B, K, -1), dim=-1).view(B, K, H, W)

        # Compute expected coordinates
        exp_x = torch.sum(x * heatmaps, dim=(2, 3))  # [B, K]
        exp_y = torch.sum(y * heatmaps, dim=(2, 3))  # [B, K]

        coords = torch.stack([exp_x, exp_y], dim=2)  # [B, K, 2]

        return coords
    
    
@MODELS.register_module()
class LineConstraintLoss(nn.Module):
    """确保特定线段经过其他线段中点的损失函数。

    例如，确保线段 (4,5) 经过线段 (0,1) 的中点，
    线段 (6,7) 经过线段 (2,3) 的中点。

    Args:
        constraints (List[Tuple[Tuple[int, int], Tuple[int, int]]]): 
            一个列表，每个元素是一个元组，包含两个元组：
            - 第一个元组是定义目标线段的关键点索引 (start, end)。
            - 第二个元组是定义参考线段的关键点索引 (ref_start, ref_end)。
        loss_weight (float): 损失的权重。默认值为 1.0。
    """
    def __init__(self, 
                 constraints: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                 loss_weight: float = 1.0):
        super().__init__()
        self.constraints = constraints
        self.loss_weight = loss_weight
    
    def forward(self, 
                output: Tensor, 
                target: Tensor = None, 
                target_weights: Tensor = None, 
                mask: Tensor = None) -> Tensor:
        """
        Args:
            output (Tensor): 预测的热图，形状为 [B, K, H, W]。
            target (Tensor, optional): 目标热图，不在此损失中使用。
            target_weights (Tensor, optional): 目标权重，不在此损失中使用。
            mask (Tensor, optional): 掩码，不在此损失中使用。
            
        Returns:
            Tensor: 计算得到的损失值。
        """
        # 提取预测的关键点坐标
        pred_coords = self._extract_coordinates(output)  # [B, K, 2]
        
        loss = 0.0
        count = 0
        for constraint in self.constraints:
            (line_start_idx, line_end_idx), (ref_line_start_idx, ref_line_end_idx) = constraint
            # 获取目标线段的起点和终点坐标
            line_start = pred_coords[:, line_start_idx, :]  # [B, 2]
            line_end = pred_coords[:, line_end_idx, :]      # [B, 2]
            # 获取参考线段的起点和终点坐标
            ref_start = pred_coords[:, ref_line_start_idx, :]  # [B, 2]
            ref_end = pred_coords[:, ref_line_end_idx, :]      # [B, 2]
            # 计算参考线段的中点
            ref_mid = (ref_start + ref_end) / 2              # [B, 2]
            # 计算中点到目标线段的距离
            distance = self._point_to_line_distance(ref_mid, line_start, line_end)  # [B]
            # 计算距离的平方作为损失
            loss += torch.mean(distance ** 2)
            count += 1
        if count > 0:
            loss = loss / count
        return loss * self.loss_weight
    
    def _extract_coordinates(self, heatmaps: Tensor) -> Tensor:
        """使用软-argmax从热图中提取关键点坐标。

        Args:
            heatmaps (Tensor): 热图，形状为 [B, K, H, W]。

        Returns:
            Tensor: 关键点坐标，形状为 [B, K, 2]，格式为 (x, y)。
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        dtype = heatmaps.dtype
        
        # 创建坐标网格
        y = torch.linspace(0, H - 1, H, device=device).view(1, 1, H, 1).expand(B, K, H, W)
        x = torch.linspace(0, W - 1, W, device=device).view(1, 1, 1, W).expand(B, K, H, W)
        
        # 对热图应用 softmax
        heatmaps = F.softmax(heatmaps.view(B, K, -1), dim=-1).view(B, K, H, W)
        
        # 计算期望坐标
        exp_x = torch.sum(x * heatmaps, dim=(2, 3))  # [B, K]
        exp_y = torch.sum(y * heatmaps, dim=(2, 3))  # [B, K]
        
        coords = torch.stack([exp_x, exp_y], dim=2)  # [B, K, 2]
        return coords
    
    def _point_to_line_distance(self, point: Tensor, 
                                line_start: Tensor, 
                                line_end: Tensor) -> Tensor:
        """计算点到直线的距离。

        Args:
            point (Tensor): 点的坐标，形状为 [B, 2]。
            line_start (Tensor): 直线起点坐标，形状为 [B, 2]。
            line_end (Tensor): 直线终点坐标，形状为 [B, 2]。

        Returns:
            Tensor: 点到直线的距离，形状为 [B]。
        """
        # 直线向量
        delta = line_end - line_start  # [B, 2]
        # 计算分子部分 |(y2 - y1)x - (x2 - x1)y + x2y1 - y2x1|
        numerator = torch.abs((delta[:, 1] * point[:, 0]) - (delta[:, 0] * point[:, 1]) 
                              + (line_end[:, 0] * line_start[:, 1]) - (line_end[:, 1] * line_start[:, 0]))  # [B]
        # 计算分母部分 √((y2 - y1)^2 + (x2 - x1)^2)，并防止除以零
        denominator = torch.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2) + 1e-6  # [B]
        distance = numerator / denominator  # [B]
        return distance


@MODELS.register_module()
class MidpointHeatmapMSELoss(nn.Module):
    """MSE Loss for Midpoint Heatmaps based on Skeleton Information.

    This loss computes the Mean Squared Error between the predicted and ground truth
    midpoint heatmaps generated directly from endpoint heatmaps based on the given skeleton_info.

    Args:
        skeleton_info (List[Tuple[int, int]]): List of tuples defining which keypoints form a skeleton line.
            Example: [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        sigma (float): Standard deviation for the Gaussian kernel used in combining heatmaps. Defaults to 1.0
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """
    
    def __init__(self,
                 skeleton_info: List[Tuple[int, int]],
                 sigma: float = 1.0,
                 loss_weight: float = 1.0):
        super().__init__()
        self.skeleton_info = skeleton_info
        self.sigma = sigma
        self.loss_weight = loss_weight

    def forward(self,
                output: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            output (Tensor): The predicted keypoint heatmaps with shape [B, K, H, W]
            target (Tensor): The ground truth midpoint heatmaps with shape [B, L, H, W]
            mask (Tensor, optional): The mask of valid heatmap pixels in shape [B, L, H, W] or [B, 1, H, W].
                                      If None, no mask is applied. Defaults to None.

        Returns:
            Tensor: The calculated loss.
        """
        # 生成预测的中点热图
        pred_mid_heatmaps = self._generate_midheatmaps(output)  # [B, L, H, W]
        target_mid_heatmaps = self._generate_midheatmaps(target)

        # 应用mask（如果提供）
        # if mask is not None:
        #     pred_mid_heatmaps = pred_mid_heatmaps * mask
        #     target_mid_heatmaps = target_mid_heatmaps * mask

        # 计算MSE Loss
        loss = F.mse_loss(pred_mid_heatmaps, target_mid_heatmaps, reduction='mean')

        return loss * self.loss_weight

    def _generate_midheatmaps(self, keypoint_heatmaps: Tensor) -> Tensor:
        """
        Generate predicted midpoint heatmaps directly from endpoint heatmaps.

        Args:
            keypoint_heatmaps (Tensor): Keypoint heatmaps with shape [B, K, H, W]

        Returns:
            Tensor: Predicted midpoint heatmaps with shape [B, L, H, W]
        """
        B, K, H, W = keypoint_heatmaps.shape
        L = len(self.skeleton_info)
        device = keypoint_heatmaps.device
        dtype = keypoint_heatmaps.dtype

        # 初始化中点热图
        pred_midheatmaps = torch.zeros((B, L, H, W), device=device, dtype=dtype)

        for idx, (kp1, kp2) in enumerate(self.skeleton_info):
            # 获取两个关键点的热图
            heatmap1 = keypoint_heatmaps[:, kp1, :, :]  # [B, H, W]
            heatmap2 = keypoint_heatmaps[:, kp2, :, :]  # [B, H, W]

            # 使用卷积生成中点热图
            # 定义一个 3x3 的均值滤波核
            kernel = torch.ones((1, 1, 3, 3), device=device, dtype=dtype) / 9.0
            heatmap1_blur = F.conv2d(heatmap1.unsqueeze(1), kernel, padding=1).squeeze(1)  # [B, H, W]
            heatmap2_blur = F.conv2d(heatmap2.unsqueeze(1), kernel, padding=1).squeeze(1)  # [B, H, W]

            # 相乘得到中点的潜在区域
            mid_heatmap = heatmap1_blur * heatmap2_blur  # [B, H, W]

            # 应用高斯模糊
            mid_heatmap = TF.gaussian_blur(mid_heatmap.unsqueeze(1), kernel_size=(3, 3), sigma=self.sigma).squeeze(1)  # [B, H, W]

            # 归一化中点热图
            max_vals = mid_heatmap.view(B, -1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).expand_as(mid_heatmap)
            mid_heatmap = mid_heatmap / (max_vals + 1e-6)  # [B, H, W]

            # 存储到中点热图张量
            pred_midheatmaps[:, idx, :, :] = mid_heatmap

        return pred_midheatmaps
