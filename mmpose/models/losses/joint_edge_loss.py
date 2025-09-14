import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


def sobel_edge(mask):
    # Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(mask.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(mask.device)
    
    # 计算梯度
    edge_x = F.conv2d(mask, sobel_x, padding=1)
    edge_y = F.conv2d(mask, sobel_y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2)
    return edge


@MODELS.register_module()
class JointEdgeLoss(nn.Module):
    def __init__(self, mask_loss_weight=1.0, keypoint_loss_weight=0.5):
        super(JointEdgeLoss, self).__init__()
        self.mask_loss_weight = mask_loss_weight
        self.keypoint_loss_weight = keypoint_loss_weight

    def forward(self, pred_keypoints, target_keypoints, pred_seg, target_seg):
        # 边缘增强损失
        target_edge = sobel_edge(target_seg)
        pred_edge = sobel_edge(pred_seg)
        edge_loss = F.mse_loss(pred_edge, target_edge)
        
        # 关键点距离损失：惩罚远离边缘的关键点
        distance_map = F.conv2d(pred_edge, torch.ones(1, 1, 3, 3).to(pred_seg.device), padding=1)
        keypoint_distances = F.grid_sample(distance_map, pred_keypoints.unsqueeze(0).unsqueeze(0), align_corners=False)
        constraint_loss = torch.mean(keypoint_distances)
        
        # 联合损失
        total_loss = self.mask_loss_weight * edge_loss + self.keypoint_loss_weight * constraint_loss
        return total_loss
