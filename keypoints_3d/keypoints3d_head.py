# coding: utf-8
'''
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

from typing import Dict, List
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import json
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, interpolate
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from typing import Tuple

_TOTAL_SKIPPED = 0


ROI_KEYPOINTS3D_HEAD_REGISTRY = Registry("ROI_KEYPOINT3D_V4_HEAD")
ROI_KEYPOINTS3D_HEAD_REGISTRY.__doc__ = """
Registry for point heads, which makes prediction for a given set of per-point features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_KEYPOINTS3D_HEAD_REGISTRY.register()
class StandardKeypoints3DHead(nn.Module):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super().__init__()

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        up_scale      = 2
        conv_dims     = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.CONV_DIMS
        num_keypoints = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.NUM_KEYPOINTS
        in_channels   = input_shape.channels
    
        self.weight_kps2d_hard   = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_2DKEYPOINTS
        self.weight_kps2d_soft = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_2DKEYPOINTS_SOFT
        self.weight_kps3d_hard = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_3D_HARD
        self.weight_kps3d_soft = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_3D_SOFT
        self.normalize_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.num_keypoints                  = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.NUM_KEYPOINTS
        batch_size_per_image                = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        positive_sample_fraction            = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        # fmt: on
        self.normalizer_per_img = (
            self.num_keypoints * batch_size_per_image * positive_sample_fraction
        )
        self.coco_to_h36_match = [0, 12, 14, 16, 11, 13, 15, 0, 0, 0, 0, 5, 7, 9, 6, 8, 10]
        with open(cfg.DATASETS.H36_STATS_PATH) as f:
            h36stats = json.load(f)
            mu2d = torch.from_numpy(
                np.array(h36stats["data_mean_2d"])[h36stats["dim_to_use_2d"]]
            ).float()
            std2d = torch.from_numpy(
                np.array(h36stats["data_std_2d"])[h36stats["dim_to_use_2d"]]
            ).float()
            mu3d = torch.from_numpy(np.array(h36stats["data_mean_3d"])).float()
            std3d = torch.from_numpy(np.array(h36stats["data_std_3d"])).float()
            dimuse3d = torch.from_numpy(np.array(h36stats["dim_to_use_3d"])).float()
            self.h36stats = {
                "mu2d": mu2d,
                "std2d": std2d,
                "mu3d": mu3d,
                "std3d": std3d,
                "dimuse3d": dimuse3d,
            }
        # fmt: on
        self.blocks  = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.blocks.append(module)
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.score_lowres_soft = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )

        self.nw2Dto3D_hard = self.make2Dto3DNetwork()
        self.nw2Dto3D_soft = self.make2Dto3DNetwork()

        self.up_scale = up_scale
        for name, param in self.named_parameters():
            if "batch_norm1" in name or "batch_norm2" in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def make2Dto3DNetwork(self):
        class Linear(nn.Module):
            def __init__(self, linear_size, p_dropout=0.5):
                super(Linear, self).__init__()
                self.l_size = linear_size

                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(p_dropout)

                self.w1 = nn.Linear(self.l_size, self.l_size)
                self.batch_norm1 = nn.BatchNorm1d(self.l_size)

                self.w2 = nn.Linear(self.l_size, self.l_size)
                self.batch_norm2 = nn.BatchNorm1d(self.l_size)

            def forward(self, x):
                y = self.w1(x)
                y = self.batch_norm1(y)
                y = self.relu(y)
                y = self.dropout(y)
                y = self.w2(y)
                y = self.batch_norm2(y)
                y = self.relu(y)
                y = self.dropout(y)
                out = x + y
                return out

        class LinearModel(nn.Module):
            def __init__(self, linear_size=1024, num_stage=2, p_dropout=0.5):
                super(LinearModel, self).__init__()

                self.linear_size = linear_size
                self.p_dropout = p_dropout
                self.num_stage = num_stage

                # 2d joints
                self.input_size = 16 * 2
                # 3d joints
                self.output_size = 16 * 3

                # process input to linear size
                self.w1 = nn.Linear(self.input_size, self.linear_size)
                self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

                self.linear_stages = []
                for l in range(num_stage):
                    self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
                self.linear_stages = nn.ModuleList(self.linear_stages)

                # post processing
                self.w2 = nn.Linear(self.linear_size, self.output_size)

                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(self.p_dropout)

            def forward(self, x):
                # pre-processing
                y = self.w1(x)
                y = self.batch_norm1(y)
                y = self.relu(y)
                y = self.dropout(y)
                # linear layers
                for i in range(self.num_stage):
                    y = self.linear_stages[i](y)
                y = self.w2(y)
                return y

        return LinearModel()

    def keypoint_rcnn_inference(self, hard_x, soft_x, pred_instances):
        """
        Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
            and add it to the `pred_instances` as a `pred_keypoints` field.

        Args:
            pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length of
            the keypoint heatmap. The values are spatial logits.
            pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

        Returns:
            None. Each element in pred_instances will contain an extra "pred_keypoints" field.
                The field is a tensor of shape (#instance, K, 3) where the last
                dimension corresponds to (x, y, score).
                The scores are larger than 0.
        """
        # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
        # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
        bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)
        num_instances_per_image = [len(i) for i in pred_instances]
        # pred_keypoint_logits_hard = pred_keypoint_logits["hard"]
        # pred_keypoint_logits_soft = pred_keypoint_logits["soft"]

        pred_keypoint_logits = (hard_x + soft_x) / 2.0
        keypoint_results = heatmaps_to_keypoints(
            pred_keypoint_logits.detach(), bboxes_flat.detach()
        )

        keypoint_results_hard = heatmaps_to_keypoints(hard_x.detach(), bboxes_flat.detach())
        keypoint_results_soft = heatmaps_to_keypoints(soft_x.detach(), bboxes_flat.detach())

        keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)
        keypoint_results_hard = keypoint_results_hard[:, :, [0, 1, 3]].split(
            num_instances_per_image, dim=0
        )
        keypoint_results_soft = keypoint_results_soft[:, :, [0, 1, 3]].split(
            num_instances_per_image, dim=0
        )

        for keypoint_results_per_image, instances_per_image in zip(
            keypoint_results, pred_instances
        ):
            # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
            instances_per_image.pred_keypoints = keypoint_results_per_image

        # Do 2D to 3D lifting and store the 3D results
        for (
            keypoint_results_per_image_hard,
            keypoint_results_per_image_soft,
            instances_per_image,
        ) in zip(keypoint_results_hard, keypoint_results_soft, pred_instances):

            poses_h36_2d_hard, poses_h36_2d_soft = [], []
            poses_h36_2d_hard_allkps, poses_h36_2d_soft_allkps = [], []
            for kps_hard, kps_soft in zip(
                keypoint_results_per_image_hard, keypoint_results_per_image_soft
            ):
                pose_h36_2d_allkps_hard = self.COCOtoH36(kps_hard)
                pose_h36_2d_allkps_soft = self.COCOtoH36(kps_soft)

                # remove the neck (root point)
                pose_h36_2d_hard = torch.cat(
                    (pose_h36_2d_allkps_hard[0:8, 0:2], pose_h36_2d_allkps_hard[9:, 0:2]), dim=0
                ).reshape(-1)
                pose_h36_2d_soft = torch.cat(
                    (pose_h36_2d_allkps_soft[0:8, 0:2], pose_h36_2d_allkps_soft[9:, 0:2]), dim=0
                ).reshape(-1)

                pose_h36_2d_hard = (
                    pose_h36_2d_hard - self.h36stats["mu2d"].to(pose_h36_2d_hard.device)
                ) / self.h36stats["std2d"].to(pose_h36_2d_hard.device)

                pose_h36_2d_soft = (
                    pose_h36_2d_soft - self.h36stats["mu2d"].to(pose_h36_2d_soft.device)
                ) / self.h36stats["std2d"].to(pose_h36_2d_soft.device)

                poses_h36_2d_hard.append(pose_h36_2d_hard)
                poses_h36_2d_soft.append(pose_h36_2d_soft)

                poses_h36_2d_hard_allkps.append(pose_h36_2d_allkps_hard)
                poses_h36_2d_soft_allkps.append(pose_h36_2d_allkps_soft)

            if poses_h36_2d_hard and poses_h36_2d_soft:
                poses_h36_2d_hard = torch.stack(poses_h36_2d_hard, dim=0)
                poses_h36_2d_soft = torch.stack(poses_h36_2d_soft, dim=0)
                poses_h36_3d_hard = self.nw2Dto3D_hard(poses_h36_2d_hard)
                poses_h36_3d_soft = self.nw2Dto3D_soft(poses_h36_2d_soft)
                instances_per_image.pred_keypoints_h36_3d = (
                    poses_h36_3d_hard + poses_h36_3d_soft
                ) / 2.0

                # add the 2d points as well
                poses_h36_2d_all_hard = torch.stack(poses_h36_2d_hard_allkps, dim=0)
                poses_h36_2d_all_soft = torch.stack(poses_h36_2d_soft_allkps, dim=0)
                instances_per_image.pred_keypoints_h36_2d = (
                    poses_h36_2d_all_hard + poses_h36_2d_all_soft
                ) / 2.0
            else:
                instances_per_image.pred_keypoints_h36_2d = torch.zeros(size=(0, 17, 3))
                instances_per_image.pred_keypoints_h36_3d = torch.zeros(size=(0, 48))

    def COCOtoH36(self, pose):
        # 0 represent not match
        _pose = pose[self.coco_to_h36_match]
        # pelvis is the mid point of lhip and rhip
        _pose[0] = (pose[11] + pose[12]) / 2.0
        # neck is the mid point of shoulder
        _pose[8] = (pose[5] + pose[6]) / 2.0
        # throax is the mid point of neck and pelvis
        _pose[7] = (_pose[0] + _pose[8]) / 2.0
        # head top are the mid points of ears
        _pose[9] = (pose[3] + pose[4]) / 2.0
        _pose[10] = (pose[3] + pose[4]) / 2.0

        return _pose

    def keypoint_rcnn_loss(
        self, pred_keypoint_logits_hard, pred_keypoint_logits_soft, instances, normalizer
    ):
        """
        Arguments:
            pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
                of instances in the batch, K is the number of keypoints, and S is the side length
                of the keypoint heatmap. The values are spatial logits.
            instances (list[Instances]): A list of M Instances, where M is the batch size.
                These instances are predictions from the model
                that are in 1:1 correspondence with pred_keypoint_logits.
                Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
                instance.
            normalizer (float): Normalize the loss by this amount.
                If not specified, we normalize by the number of visible keypoints in the minibatch.

        Returns a scalar tensor containing the loss.
        """
        heatmaps = []
        valid = []

        heatmaps_soft = []
        valid_soft = []
        conf_soft = []
        reg_input_2d_soft = []
        reg_input_2d_hard = []
        reg_target_3d_soft = []
        reg_target_3d_hard = []

        # pred_keypoint_logits = (pred_keypoint_logits_hard + pred_keypoint_logits_soft) / 2.0

        # calculate the loss for the hard and soft keypoints
        keypoint_side_len = pred_keypoint_logits_hard.shape[2]
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            keypoints = instances_per_image.gt_keypoints
            heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
                instances_per_image.proposal_boxes.tensor, keypoint_side_len
            )
            heatmaps.append(heatmaps_per_image.view(-1))
            valid.append(valid_per_image.view(-1))

            # heatmaps for the soft keypoints
            heatmaps_per_image_soft, valid_per_image_soft, conf_per_image_soft = self.soft_keypoints_to_heatmap(
                instances_per_image.gt_keypoints_soft.tensor,
                instances_per_image.proposal_boxes.tensor,
                keypoint_side_len,
            )
            heatmaps_soft.append(heatmaps_per_image_soft.view(-1))
            valid_soft.append(valid_per_image_soft.view(-1))
            conf_soft.append(conf_per_image_soft.view(-1))

            reg_input_2d_soft.append(instances_per_image.gt_keypoints2d_soft)
            reg_input_2d_hard.append(instances_per_image.gt_keypoints2d_hard)
            reg_target_3d_soft.append(instances_per_image.gt_keypoints3d_soft)
            reg_target_3d_hard.append(instances_per_image.gt_keypoints3d_hard)

        if len(heatmaps):
            keypoint_targets = cat(heatmaps, dim=0)
            valid = cat(valid, dim=0).to(dtype=torch.uint8)
            valid = torch.nonzero(valid).squeeze(1)

        if len(heatmaps_soft):
            keypoint_targets_soft = cat(heatmaps_soft, dim=0)
            conf_targets = cat(conf_soft, dim=0)
            valid_soft = cat(valid_soft, dim=0).to(dtype=torch.uint8)
            valid_soft = torch.nonzero(valid_soft).squeeze(1)

        if len(reg_target_3d_hard):
            reg_target_3d_hard = cat(reg_target_3d_hard, dim=0)
            reg_input_2d_hard = cat(reg_input_2d_hard, dim=0)
            indices = torch.nonzero(reg_input_2d_hard.sum(dim=1).to(dtype=torch.uint8))
            reg_target_3d_hard = reg_target_3d_hard[indices.squeeze()]
            reg_input_2d_hard = reg_input_2d_hard[indices.squeeze()]

        if len(reg_target_3d_soft):
            reg_target_3d_soft = cat(reg_target_3d_soft, dim=0)
            reg_input_2d_soft = cat(reg_input_2d_soft, dim=0)
            indices = torch.nonzero(reg_input_2d_soft.sum(dim=1).to(dtype=torch.uint8))
            reg_target_3d_soft = reg_target_3d_soft[indices.squeeze()]
            reg_input_2d_soft = reg_input_2d_soft[indices.squeeze()]            

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if len(heatmaps) == 0 or valid.numel() == 0:
            global _TOTAL_SKIPPED
            _TOTAL_SKIPPED += 1
            storage = get_event_storage()
            storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
            return (
                pred_keypoint_logits_hard.sum() * 0,
                pred_keypoint_logits_soft.sum() * 0,
                pred_keypoint_logits_hard.sum() * 0,
                pred_keypoint_logits_soft.sum() * 0,
            )

        N, K, H, W = pred_keypoint_logits_hard.shape
        pred_keypoint_logits_hard = pred_keypoint_logits_hard.view(N * K, H * W)

        keypoint_loss = F.cross_entropy(
            pred_keypoint_logits_hard[valid], keypoint_targets[valid], reduction="sum"
        )

        # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
        if normalizer is None:
            new_normalizer = valid.numel()
        else:
            new_normalizer = normalizer
        keypoint_loss /= new_normalizer

        # calculate the cross entropy loss from the soft labels
        N, K, H, W = pred_keypoint_logits_soft.shape
        pred_keypoint_logits_soft = pred_keypoint_logits_soft.view(N * K, H * W)

        keypoint_loss_soft = F.cross_entropy(
            pred_keypoint_logits_soft[valid_soft] * conf_targets[valid_soft].unsqueeze(1),
            keypoint_targets_soft[valid_soft],
            reduction="sum",
        )
        if normalizer is None:
            new_normalizer = valid_soft.numel()
        else:
            new_normalizer = normalizer
        keypoint_loss_soft /= new_normalizer

        if reg_input_2d_hard.shape[0] == 0:
            loss_3d_hard = pred_keypoint_logits_hard.sum() * 0
        else:
            out3d = self.nw2Dto3D_hard(reg_input_2d_hard)
            loss_3d_hard = F.smooth_l1_loss(out3d, reg_target_3d_hard, reduction="mean")

        if reg_input_2d_soft.shape[0] == 0:
            loss_3d_soft = pred_keypoint_logits_soft.sum() * 0
        else:
            out3d = self.nw2Dto3D_soft(reg_input_2d_soft)
            loss_3d_soft = F.smooth_l1_loss(out3d, reg_target_3d_soft, reduction="mean")

        return keypoint_loss, keypoint_loss_soft, loss_3d_hard, loss_3d_soft

    def soft_keypoints_to_heatmap(
        self, keypoints: torch.Tensor, rois: torch.Tensor, heatmap_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode keypoint locations into a target heatmap for use in SoftmaxWithLoss across space.

        Maps keypoints from the half-open interval [x1, x2) on continuous image coordinates to the
        closed interval [0, heatmap_size - 1] on discrete image coordinates. We use the
        continuous-discrete conversion from Heckbert 1990 ("What is the coordinate of a pixel?"):
        d = floor(c) and c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.

        Arguments:
            keypoints: tensor of keypoint locations in of shape (N, K, 3).
            rois: Nx4 tensor of rois in xyxy format
            heatmap_size: integer side length of square heatmap.

        Returns:
            heatmaps: A tensor of shape (N, K) containing an integer spatial label
                in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
            valid: A tensor of shape (N, K) containing whether each keypoint is in
                the roi or not.
        """

        if rois.numel() == 0:
            return rois.new().long(), rois.new().long()
        offset_x = rois[:, 0]
        offset_y = rois[:, 1]
        scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
        scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

        offset_x = offset_x[:, None]
        offset_y = offset_y[:, None]
        scale_x = scale_x[:, None]
        scale_y = scale_y[:, None]

        x = keypoints[..., 0]
        y = keypoints[..., 1]

        x_boundary_inds = x == rois[:, 2][:, None]
        y_boundary_inds = y == rois[:, 3][:, None]

        x = (x - offset_x) * scale_x
        x = x.floor().long()
        y = (y - offset_y) * scale_y
        y = y.floor().long()

        x[x_boundary_inds] = heatmap_size - 1
        y[y_boundary_inds] = heatmap_size - 1

        valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
        vis = keypoints[..., 2] > 0
        valid = (valid_loc & vis).long()

        lin_ind = y * heatmap_size + x
        heatmaps = lin_ind * valid

        conf = keypoints[..., 2] * valid

        return heatmaps, valid, conf

    def layers(self, x):
        for layer in self.blocks:
            x = F.relu(layer(x))

        hard_x = self.score_lowres(x)
        hard_x = interpolate(
            hard_x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )

        soft_x = self.score_lowres_soft(x)
        soft_x = interpolate(
            soft_x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return hard_x, soft_x

    def forward(self, x: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Args:
            x (dict[str,Tensor]): input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        hard_x, soft_x = self.layers(x)
        if self.training:
            num_images = len(instances)
            normalizer = (
                None
                if self.normalize_by_visible_keypoints
                else num_images * self.normalizer_per_img
            )
            loss_2d_hard, loss_2d_soft, loss_3d_hard, loss_3d_soft = self.keypoint_rcnn_loss(
                hard_x, soft_x, instances, normalizer=normalizer
            )
            return {
                "loss_keypoints_2D_hard": self.weight_kps2d_hard * loss_2d_hard,
                "loss_keypoints_2D_soft": self.weight_kps2d_soft * loss_2d_soft,
                "loss_keypoints_3D_hard": self.weight_kps3d_hard * loss_3d_hard,
                "loss_keypoints_3D_soft": self.weight_kps3d_soft * loss_3d_soft,
            }
        else:
            self.keypoint_rcnn_inference(hard_x, soft_x, instances)
            return instances


def build_keypoints3d_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.NAME
    return ROI_KEYPOINTS3D_HEAD_REGISTRY.get(head_name)(cfg, input_channels)
