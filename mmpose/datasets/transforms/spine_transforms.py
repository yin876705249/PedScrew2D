# spine_transforms.py
from typing import Optional

import cv2
from .common_transforms import RandomFlip, Albumentation
from .topdown_transforms import TopdownAffine
from mmcv.image import imflip

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix


@TRANSFORMS.register_module()
class SpineRandomFlip(RandomFlip):
    """Randomly flip the image and mask."""
    def transform(self, results: dict) -> dict:
        super().transform(results)
        if 'flip' in results and results['flip']:
            # Perform flip on mask if mask exists
            if 'gt_sem_seg' in results:
                results['gt_sem_seg'] = imflip(results['gt_sem_seg'], direction=results['flip_direction'])
        return results


@TRANSFORMS.register_module()
class SpineAlbumentation(Albumentation):
    """Apply Albumentations augmentations to both image and mask."""
    def transform(self, results: dict) -> dict:
        # Map result dict to albumentations format
        results_albu = {'image': results['img']}
        if 'gt_sem_seg' in results:
            results_albu['gt_sem_seg'] = results['gt_sem_seg']

        # Apply albumentations transforms
        transformed = self.aug(**results_albu)

        # Map the albu results back to the original format
        results['img'] = transformed['image']
        if 'gt_sem_seg' in transformed:
            results['gt_sem_seg'] = transformed['gt_sem_seg']

        return results


@TRANSFORMS.register_module()
class SpineTopdownAffine(TopdownAffine):
    """Enhanced TopdownAffine that also transforms the mask along with the image."""

    def transform(self, results: dict) -> Optional[dict]:
        super().transform(results)  # Apply the original image transformations

        if 'gt_sem_seg' in results:
            w, h = self.input_size
            warp_size = (int(w), int(h))
            
            if 'bbox_rotation' in results:
                rot = results['bbox_rotation'][0]
            else:
                rot = 0.

            center = results['bbox_center'][0]
            scale = results['bbox_scale'][0]

            # Determine the appropriate warp matrix
            if self.use_udp:
                warp_mat = get_udp_warp_matrix(center, scale, rot, output_size=(w, h))
            else:
                warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))
            
            # Apply affine transformation to the mask
            results['gt_sem_seg'] = cv2.warpAffine(
                results['gt_sem_seg'], warp_mat, warp_size, flags=cv2.INTER_NEAREST)  # Use INTER_NEAREST for masks

        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(input_size={self.input_size}, use_udp={self.use_udp})'