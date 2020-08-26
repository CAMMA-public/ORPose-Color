# coding: utf-8
'''
Copyright (c) University of Strasbourg, All Rights Reserved.
'''
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.poolers import ROIPooler
from .keypoints3d_head import build_keypoints3d_head


from detectron2.modeling.roi_heads.roi_heads import (
    select_foreground_proposals,
    select_proposals_with_visible_keypoints,
)
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads


@ROI_HEADS_REGISTRY.register()
class Keypoints3DROIHeads(StandardROIHeads):
    """
    """

    def __init__(self, cfg, input_shape):
        super(Keypoints3DROIHeads, self).__init__(cfg, input_shape)
        self._init_keypoints3d_head(cfg, input_shape)

    def _init_keypoints3d_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT3D_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINTS3D_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.keypoint_head = build_keypoints3d_head(
            cfg,
            ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution),
        )

    def _forward_keypoints3d(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            # losses.update(self._forward_mask(features, proposals))
            # losses.update(self._forward_keypoint(features, proposals))
            losses.update(self._forward_keypoints3d(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            assert not self.training
            assert pred_instances[0].has("pred_boxes") and pred_instances[0].has("pred_classes")

            # pred_instances = self._forward_mask(features, pred_instances)
            # red_instances = self._forward_keypoint(features, pred_instances)
            pred_instances = self._forward_keypoints3d(features, pred_instances)

            return pred_instances, {}
