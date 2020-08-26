# coding: utf-8
'''
Copyright (c) University of Strasbourg, All Rights Reserved.
'''
import numpy as np
import json
import torch
import os
import cv2
import copy
from PIL import Image
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.structures import Keypoints, Boxes, BoxMode, Instances
from detectron2.data import MetadataCatalog


def register_datasets(cfg, args):
    """
    Register the MVOR dataset (format is same as COCO; so easy to register)
    """
    # Register the test or val dataset in the coco format
    # also incorporate the 3D ground truth keypoints
    metadata = {}
    if "mvor" in cfg.DATASETS.TEST_SET_NAME:
        dataset_dict_test = load_coco_json(
            cfg.DATASETS.TEST_SET_JSON,
            cfg.DATASETS.TEST_SET_IMGDIR,
            cfg.DATASETS.TEST_SET_NAME,
            extra_annotation_keys=[
                "keypoints3D_camma_projected_true",
                "keypoints3D_camma_projected_allkps",
                "keypoints2D_camma",
                "person_id",
            ],
        )
        DatasetCatalog.register(cfg.DATASETS.TEST_SET_NAME, lambda: dataset_dict_test)
        MetadataCatalog.get(cfg.DATASETS.TEST_SET_NAME).set(
            json_file=cfg.DATASETS.TEST_SET_JSON,
            image_root=cfg.DATASETS.TEST_SET_IMGDIR,
            evaluator_type="coco",
            **metadata
        )
    else:
        register_coco_instances(
            cfg.DATASETS.TEST_SET_NAME, {}, cfg.DATASETS.TEST_SET_JSON, cfg.DATASETS.TEST_SET_IMGDIR
        )
    MetadataCatalog.get(cfg.DATASETS.TEST_SET_NAME).keypoint_names = cfg.DATASETS.KEYPOINTS_NAMES
    MetadataCatalog.get(
        cfg.DATASETS.TEST_SET_NAME
    ).keypoint_flip_map = cfg.DATASETS.KEYPOINTS_FLIP_MAP
    if args.eval_only:
        return

    # register the train set with hard and soft labels
    dataset_dict_train = load_coco_json(
        cfg.DATASETS.TRAIN_SET_JSON,
        cfg.DATASETS.TRAIN_SET_IMGDIR,
        cfg.DATASETS.TRAIN_SET_NAME,
        extra_annotation_keys=[
            "keypoints_soft",
            "score",
            "keypoints3d_hard",
            "keypoints2d_hard",
            "keypoints3d_soft",
            "keypoints2d_soft",
        ],
    )
    DatasetCatalog.register(cfg.DATASETS.TRAIN_SET_NAME, lambda: dataset_dict_train)
    MetadataCatalog.get(cfg.DATASETS.TRAIN_SET_NAME).set(
        json_file=cfg.DATASETS.TRAIN_SET_JSON,
        image_root=cfg.DATASETS.TRAIN_SET_JSON,
        evaluator_type="coco",
        **metadata
    )

    MetadataCatalog.get(cfg.DATASETS.TRAIN_SET_NAME).keypoint_names = cfg.DATASETS.KEYPOINTS_NAMES
    MetadataCatalog.get(
        cfg.DATASETS.TRAIN_SET_NAME
    ).keypoint_flip_map = cfg.DATASETS.KEYPOINTS_FLIP_MAP


class MVORDatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.cfg = cfg
        self.keypoint_on = cfg.MODEL.KEYPOINT3D_ON
        self.kps_index = {k: v for v, k in enumerate(self.cfg.DATASETS.KEYPOINTS_NAMES)}
        if self.keypoint_on and is_train:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        self.is_test_set_camma = "mvor" in cfg.DATASETS.TEST_SET_NAME
        self.num_kps_camma3d = cfg.DATASETS.NUM_KEYPOINTS_CAMMA3D

        if not self.is_train:
            with open(cfg.DATASETS.TEST_SET_JSON) as f:
                self.intrinsic = json.load(f)["cameras_info"]["camParams"]["intrinsics"]

    def _get_scale_interp(self):
        if self.cfg.DATASETS.FIX_DOWN_SAMPLING:
            scale = self.cfg.DATASETS.DOWN_SAMPLING_FIX_SCALE
        else:
            if self.is_train:
                if torch.rand(1).item() > 0.3:
                    if torch.rand(1).item() > 0.5:
                        scale = torch.LongTensor(1).random_(8, 12).item()
                    else:
                        scale = torch.LongTensor(1).random_(2, 8).item()
                else:
                    scale = 1.0
            else:
                scale = self.cfg.DATASETS.DOWN_SAMPLING_TEST_SCALE

        scale = float(scale)
        interp_method = (
            Image.BILINEAR
            if self.cfg.DATASETS.DOWN_SAMPLING_METHOD == "bilinear"
            else Image.BICUBIC
        )
        return scale, interp_method

    def _get_cam_params(self, cam_id):
        # 1:4 encode the cam_id 001-> cam1; 002-> cam2; 003-> cam3
        fx_d, fy_d = self.intrinsic[cam_id]["focallength"]
        cx_d, cy_d = self.intrinsic[cam_id]["principalpoint"]
        return fx_d, fy_d, cx_d, cy_d

    def _transform(self, image, scale, interp_method):
        h, w, _ = image.shape
        new_h, new_w = int(h / scale), int(w / scale)
        if self.is_train:
            image, transforms = T.apply_transform_gens(
                [
                    T.RandomFlip(),
                    T.Resize(shape=(new_h, new_w), interp=interp_method),
                    T.Resize(shape=(h, w), interp=interp_method),
                ],
                image,
            )
        else:
            image, transforms = T.apply_transform_gens(
                [
                    T.Resize(shape=(new_h, new_w), interp=interp_method),
                    T.Resize(shape=(h, w), interp=interp_method),
                ],
                image,
            )
        return image, transforms

    def _get_camma_gt(self, dataset_dict):
        anns2d_camma = [
            np.array(k["keypoints2D_camma"]).reshape(self.num_kps_camma3d, 3).astype(np.float32)
            for k in dataset_dict["annotations"]
        ]
        anns3d_camma_allkps = [
            np.array(k["keypoints3D_camma_projected_allkps"])
            .reshape(self.num_kps_camma3d, 4)
            .astype(np.float32)
            for k in dataset_dict["annotations"]
        ]
        anns3d_camma_true = [
            np.array(k["keypoints3D_camma_projected_true"])
            .reshape(self.num_kps_camma3d, 4)
            .astype(np.float32)
            for k in dataset_dict["annotations"]
        ]
        person_ids = [k["person_id"] for k in dataset_dict["annotations"]]
        if anns2d_camma and anns3d_camma_allkps and anns3d_camma_true:
            anns2d_camma = torch.tensor(np.stack(anns2d_camma))
            anns3d_camma_allkps = torch.tensor(np.stack(anns3d_camma_allkps))
            anns3d_camma_true = torch.tensor(np.stack(anns3d_camma_true))
            person_ids = torch.tensor(person_ids)
            return anns2d_camma, anns3d_camma_allkps, anns3d_camma_true, person_ids
        else:
            return None, None, None, None

    def transform_instance_annotations(
        self, annotation, transforms, image_size, *, keypoint_hflip_indices=None
    ):
        """
        Apply transforms to box, keypoints annotations of a single instance.

        It will use `transforms.apply_box` for the box, and
        `transforms.apply_coords` for segmentation polygons & keypoints.
        If you need anything more specially designed for each data structure,
        you'll need to implement your own version of this function or the transforms.

        Args:
            annotation (dict): dict of instance annotations for a single instance.
                It will be modified in-place.
            transforms (TransformList):
            image_size (tuple): the height, width of the transformed image
            keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

        Returns:
            dict:
                the same input dict with fields "bbox", "segmentation", "keypoints"
                transformed according to `transforms`.
                The "bbox_mode" field will be set to XYXY_ABS.
        """
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        if "keypoints" in annotation:
            keypoints = utils.transform_keypoint_annotations(
                annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
            )
            annotation["keypoints"] = keypoints

        if "keypoints_soft" in annotation:
            keypoints = utils.transform_keypoint_annotations(
                annotation["keypoints_soft"], transforms, image_size, keypoint_hflip_indices
            )
            annotation["keypoints_soft"] = keypoints

        return annotation

    def annotations_to_instances(self, annos, image_size):
        """
        Create an :class:`Instances` object used by the models,
        from instance annotations in the dataset dict.

        Args:
            annos (list[dict]): a list of instance annotations in one image, each
                element for one instance.
            image_size (tuple): height, width

        Returns:
            Instances:
                It will contain fields "gt_boxes", "gt_classes",
                "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
                This is the format that builtin models expect.
        """
        boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        target = Instances(image_size)
        boxes = target.gt_boxes = Boxes(boxes)
        boxes.clip(image_size)

        classes = [obj["category_id"] for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        if len(annos) and "keypoints" in annos[0]:
            kpts = [obj.get("keypoints", []) for obj in annos]
            target.gt_keypoints = Keypoints(kpts)

        if len(annos) and "keypoints_soft" in annos[0]:
            kpts = [obj.get("keypoints_soft", []) for obj in annos]
            target.gt_keypoints_soft = Keypoints(kpts)

        if len(annos) and "keypoints3d_hard" in annos[0]:
            kpts = np.array([obj.get("keypoints3d_hard", []) for obj in annos])
            target.gt_keypoints3d_hard = torch.as_tensor(kpts, dtype=torch.float32) 

        if len(annos) and "keypoints2d_hard" in annos[0]:
            kpts = np.array([obj.get("keypoints2d_hard", []) for obj in annos])
            target.gt_keypoints2d_hard = torch.as_tensor(kpts, dtype=torch.float32)
        
        if len(annos) and "keypoints3d_soft" in annos[0]:
            kpts = np.array([obj.get("keypoints3d_soft", []) for obj in annos])
            target.gt_keypoints3d_soft = torch.as_tensor(kpts, dtype=torch.float32) 

        if len(annos) and "keypoints2d_soft" in annos[0]:
            kpts = np.array([obj.get("keypoints2d_soft", []) for obj in annos])
            target.gt_keypoints2d_soft = torch.as_tensor(kpts, dtype=torch.float32)            

        return target

    def _get_cam_params(self, cam_id):
        # 1:4 encode the cam_id 001-> cam1; 002-> cam2; 003-> cam3
        fx_d, fy_d = self.intrinsic[cam_id]["focallength"]
        cx_d, cy_d = self.intrinsic[cam_id]["principalpoint"]
        return fx_d, fy_d, cx_d, cy_d

    def _update_color_anns(self, dataset_dict, image, transforms):
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        #flipped = "HFlipTransform" in str(transforms.transforms[0].__class__)

        annos = [
            self.transform_instance_annotations(
                obj, transforms, image.shape[:2], keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = self.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        # update with the 3D ground truths in camma format for the test set
        if not self.is_train and self.is_test_set_camma:
            anns2d_camma, anns3d_camma_allkps, anns3d_camma_true, person_ids = self._get_camma_gt(
                dataset_dict
            )
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        scale, interp_method = self._get_scale_interp()
        image, transforms = self._transform(image, scale, interp_method)
        dataset_dict = self._update_color_anns(dataset_dict, image, transforms)

        # Add ground truth for the camma-mvor during testing
        if not self.is_train and self.is_test_set_camma:
            if (
                anns2d_camma is not None
                and anns3d_camma_allkps is not None
                and anns3d_camma_true is not None
            ):
                dataset_dict["instances"]._fields["anns2d_camma"] = anns2d_camma
                dataset_dict["instances"]._fields["anns3d_camma_allkps"] = anns3d_camma_allkps
                dataset_dict["instances"]._fields["anns3d_camma_true"] = anns3d_camma_true
                dataset_dict["instances"]._fields["person_ids_camma"] = person_ids
            fx_d, fy_d, cx_d, cy_d = self._get_cam_params(
                int(str(dataset_dict["image_id"])[1:4]) - 1
            )
            dataset_dict["instances"]._fields["focal_length"] = torch.tensor(
                [fx_d, fy_d], dtype=torch.float32
            )
            dataset_dict["instances"]._fields["principal_point"] = torch.tensor(
                [cx_d, cy_d], dtype=torch.float32
            )
        return dataset_dict