from mmengine.evaluator import BaseMetric
import numpy as np
from mmpose.registry import METRICS

@METRICS.register_module()
class ScrewMetric(BaseMetric):
    def __init__(self, skeleton_info, sigmas, **kwargs):
        super().__init__(**kwargs)
        self.skeleton_info = skeleton_info
        self.sigmas = np.array(sigmas) / 10
        self.results = []


    def dice_coefficient(self, pred_mask, gt_mask):
        """Compute the Dice coefficient between prediction and ground truth masks."""
        smooth = 1.0  # Add smooth to avoid division by zero
        pred_mask_flat = pred_mask.flatten()
        gt_mask_flat = gt_mask.flatten()

        intersection = np.sum(pred_mask_flat * gt_mask_flat)
        union = np.sum(pred_mask_flat) + np.sum(gt_mask_flat)

        dice = (2. * intersection + smooth) / (union + smooth)
        return dice


    def compute_angle_error(self, pred_segment, gt_segment):
        pred_vector = pred_segment[1] - pred_segment[0]
        gt_vector = gt_segment[1] - gt_segment[0]
        pred_vector = pred_vector / np.linalg.norm(pred_vector)
        gt_vector = gt_vector / np.linalg.norm(gt_vector)
        dot_product = np.clip(np.dot(pred_vector, gt_vector), -1.0, 1.0)
        angle_error = np.arccos(dot_product) * (180.0 / np.pi)
        return angle_error

    def point_to_segment_distance(self, point, segment):
        seg_start, seg_end = segment
        seg_vector = seg_end - seg_start
        point_vector = point - seg_start
        proj_length = np.dot(point_vector, seg_vector) / np.dot(seg_vector, seg_vector)
        proj_length = np.clip(proj_length, 0, 1)
        nearest_point = seg_start + proj_length * seg_vector
        return np.linalg.norm(point - nearest_point)

    # def compute_mnd(self, pred_segment, gt_segment):
    #     distances = []
    #     for point in [pred_segment[0], pred_segment[1]]:
    #         distances.append(self.point_to_segment_distance(point, gt_segment))
    #     for point in [gt_segment[0], gt_segment[1]]:
    #         distances.append(self.point_to_segment_distance(point, pred_segment))

    #     gt_length = np.linalg.norm(gt_segment[1] - gt_segment[0])
    #     mean_distance = np.mean(distances)
    #     normalized_mean_distance = mean_distance / gt_length if gt_length != 0 else float('nan')

    #     return normalized_mean_distance

    # def compute_length_error(self, pred_segment, gt_segment):
    #     pred_length = np.linalg.norm(pred_segment[1] - pred_segment[0])
    #     gt_length = np.linalg.norm(gt_segment[1] - gt_segment[0])
    #     length_error = np.abs(pred_length - gt_length) / gt_length if gt_length != 0 else float('nan')
    #     return length_error
    
    def compute_mnd(self, pred_segment, gt_segment, k=1.0):
        distances = []
        for point in [pred_segment[0], pred_segment[1]]:
            distances.append(self.point_to_segment_distance(point, gt_segment))
        for point in [gt_segment[0], gt_segment[1]]:
            distances.append(self.point_to_segment_distance(point, pred_segment))

        gt_length = np.linalg.norm(gt_segment[1] - gt_segment[0])
        if gt_length == 0:
            return float('nan')

        # Normalize each distance by gt_length and apply negative exponential
        normalized_distances = np.array(distances) / gt_length
        exp_scores = np.exp(-normalized_distances / k)

        # Take the mean of the exponential scores
        mnd_score = np.mean(exp_scores)
        return mnd_score

    def compute_length_error(self, pred_segment, gt_segment, k=1.0):
        pred_length = np.linalg.norm(pred_segment[1] - pred_segment[0])
        gt_length = np.linalg.norm(gt_segment[1] - gt_segment[0])
        if gt_length == 0:
            return float('nan')

        length_error = np.abs(pred_length - gt_length) / gt_length

        # Normalize to 0-1 using negative exponential
        length_error_score = np.exp(-length_error / k)
        return length_error_score

    def compute_oks(self, pred_keypoints, gt_keypoints, visible, image_area):
        """Compute Object Keypoint Similarity (OKS) for one set of keypoints."""
        if self.sigmas is None:
            num_keypoints = pred_keypoints.shape[0]
            self.sigmas = np.full((num_keypoints,), 0.25) / 10

        variances = (self.sigmas * 2) ** 2

        dx = pred_keypoints[..., 0] - gt_keypoints[..., 0]
        dy = pred_keypoints[..., 1] - gt_keypoints[..., 1]
        e = (dx**2 + dy**2) / variances / (image_area + np.spacing(1)) / 2

        oks = np.zeros(len(e))
        for i in range(len(e)):
            if visible[i]:
                oks[i] = np.exp(-e[i])
            else:
                oks[i] = 0.0

        return oks.mean()

    def compute_oks_l(self, pred_keypoints, gt_keypoints, visible, sigmas=None):
        """Compute OKS normalized by the actual length of each segment."""
        if sigmas is None:
            num_keypoints = pred_keypoints.shape[0]
            sigmas = np.full((num_keypoints,), 0.25) / 10

        variances = (sigmas * 2) ** 2
        oks_l_scores = []

        for start_idx, end_idx in self.skeleton_info:
            if visible[start_idx] and visible[end_idx]:
                segment_length = np.linalg.norm(gt_keypoints[end_idx] - gt_keypoints[start_idx])
                dx = pred_keypoints[start_idx, 0] - gt_keypoints[start_idx, 0]
                dy = pred_keypoints[start_idx, 1] - gt_keypoints[start_idx, 1]
                e = (dx**2 + dy**2) / variances[start_idx] / (segment_length + np.spacing(1)) / 2
                oks_l_score = np.exp(-e)
                oks_l_scores.append(oks_l_score)
            else:
                oks_l_scores.append(0.0)

        return np.mean(oks_l_scores) if oks_l_scores else 0.0
    
    def compute_sps(self, pred_segment, gt_segment, num_samples=10):
        """Calculate Segment Proximity Score (SPS) with multiple sampled points on the predicted segment."""
        seg_start, seg_end = pred_segment
        sampled_points = [seg_start + t * (seg_end - seg_start) for t in np.linspace(0, 1, num_samples)]
        distances = [self.point_to_segment_distance(point, gt_segment) for point in sampled_points]
        real_segment_length = np.linalg.norm(gt_segment[1] - gt_segment[0])
        sps = np.mean(distances) / real_segment_length if real_segment_length != 0 else float('inf')
        return sps


    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            pred_keypoints = data_sample['pred_instances']['keypoints']
            gt_keypoints = data_sample['gt_instances']['keypoints']
            visible = data_sample['gt_instances']['keypoints_visible'].astype(bool).reshape(-1, pred_keypoints.shape[1])
            bbox = data_sample['pred_instances']['bboxes'][0]
            image_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            results = {'angle_errors': [], 'mnd': [], 'length_errors': [], 'oks': [], 'oks_l': [], 'sps': []}
            for start_idx, end_idx in self.skeleton_info:
                if visible[0, start_idx] and visible[0, end_idx]:
                    pred_segment = pred_keypoints[0, [start_idx, end_idx], :]
                    gt_segment = gt_keypoints[0, [start_idx, end_idx], :]

                    angle_error = self.compute_angle_error(pred_segment, gt_segment)
                    mnd_error = self.compute_mnd(pred_segment, gt_segment)
                    length_error = self.compute_length_error(pred_segment, gt_segment)
                    sps_value = self.compute_sps(pred_segment, gt_segment)

                    results['angle_errors'].append(angle_error)
                    results['mnd'].append(mnd_error)
                    results['length_errors'].append(length_error)
                    results['sps'].append(sps_value)

            oks_value = self.compute_oks(pred_keypoints[0], gt_keypoints[0], visible[0], image_area)
            oks_l_value = self.compute_oks_l(pred_keypoints[0], gt_keypoints[0], visible[0])
            results['oks'].append(oks_value)
            results['oks_l'].append(oks_l_value)

            self.results.append(results)

    def compute_metrics(self, results):
        angle_errors = []
        mnd_errors = []
        length_errors = []
        oks_values = []
        oks_l_values = []
        sps_values = []

        for result in results:
            angle_errors.extend(result['angle_errors'])
            mnd_errors.extend(result['mnd'])
            length_errors.extend(result['length_errors'])
            oks_values.extend(result['oks'])
            oks_l_values.extend(result['oks_l'])
            sps_values.extend(result['sps'])

        mean_angle_error = np.mean(angle_errors) if angle_errors else float('nan')
        mean_mnd_error = np.mean(mnd_errors) if mnd_errors else float('nan')
        mean_length_error = np.mean(length_errors) if length_errors else float('nan')
        mean_oks = np.mean(oks_values) if oks_values else float('nan')
        mean_oks_l = np.mean(oks_l_values) if oks_l_values else float('nan')
        mean_sps = np.mean(sps_values) if sps_values else float('nan')

        # Calculate AP, AP@50, AP@75, and AR
        ap = self.calculate_ap(oks_values)
        ap_at_50 = self.calculate_ap(oks_values, thresholds=[0.5])
        ap_at_75 = self.calculate_ap(oks_values, thresholds=[0.75])
        ar = self.calculate_ar(oks_values)

        return {
            'OKS': mean_oks,
            'AP': ap,
            'AP@50': ap_at_50,
            'AP@75': ap_at_75,
            'AR': ar,
            'angle_error': mean_angle_error,
            'mnd_error': mean_mnd_error,
            'length_error': mean_length_error,
            'oks_l': mean_oks_l,
            'SPS': mean_sps
        }

    def calculate_ap(self, oks_values, thresholds=np.arange(0.5, 1.0, 0.05)):
        """Calculate Average Precision (AP) given OKS values and thresholds."""
        ap = []
        for t in thresholds:
            correct = np.sum(np.array(oks_values) > t)
            precision = correct / len(oks_values) if oks_values else 0
            ap.append(precision)
        return np.mean(ap)

    def calculate_ar(self, oks_values, thresholds=np.arange(0.5, 1.0, 0.05)):
        """Calculate Average Recall (AR) given OKS values and thresholds."""
        ar = []
        for t in thresholds:
            recall = np.sum(np.array(oks_values) > t) / len(oks_values) if oks_values else 0
            ar.append(recall)
        return np.mean(ar)
