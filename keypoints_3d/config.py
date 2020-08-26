# coding: utf-8
'''
Copyright (c) University of Strasbourg, All Rights Reserved.
'''
from detectron2.config import CfgNode as CN
def add_keypoints3d_config(cfg):
    """
    Add config for Keypoints3D.
    """

    # solver
    cfg.MODEL.ROI_HEADS.NUM_CLASSES= 1
    cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.5
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500

    cfg.MODEL.KEYPOINT3D_ON = True
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD = CN()
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.NAME = "StandardKeypoints3DHead"
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.NUM_CLASSES = 2
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.NUM_KEYPOINTS = 17  # 17 is the number of keypoints in COCO/Human3.6/MVOR
      
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_2DKEYPOINTS = 1.0
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_2DKEYPOINTS_SOFT = 0.0
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_2DKEYPOINTS_HARD = 0.0
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_3D_SOFT = 0.0
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.LOSS_WEIGHT_3D_HARD = 0.0
    cfg.MODEL.ROI_KEYPOINTS3D_HEAD.POOLER_TYPE = "ROIAlignV2"

    # dataset settings
    cfg.DATASETS.H36_STATS_PATH = ""
    cfg.DATASETS.TEST_SET_NAME = ""
    cfg.DATASETS.TEST_SET_JSON = ""
    cfg.DATASETS.TEST_SET_IMGDIR = ""

    cfg.DATASETS.NUM_KEYPOINTS_CAMMA3D = 10

    # for downsampling
    cfg.DATASETS.DOWN_SAMPLING = False
    cfg.DATASETS.FIX_DOWN_SAMPLING = True
    cfg.DATASETS.DOWN_SAMPLING_FIX_SCALE = 1.0
    cfg.DATASETS.DOWN_SAMPLING_TEST_SCALE = 12.0
    cfg.DATASETS.DOWN_SAMPLING_METHOD = "bilinear"
    cfg.DATASETS.BONE_LENGTHS_FILE = ""
    cfg.TEST.EVAL_3D = False 

    cfg.DATASETS.KEYPOINTS_NAMES = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    cfg.DATASETS.KEYPOINTS_FLIP_MAP = (
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    )
    cfg.DATASETS.KEYPOINTS_SKELTON_MAP = (
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    )
    cfg.DATASETS.SKELTON_COLORS=["m", "m", "g", "g", "r", "m", "g", "r", "m", "g", "m", "g", "r", "m", "g", "m", "g", "m", "g"]
    cfg.DATASETS.SKELTON_COLORS_TRIPLET = [
        ["r", [0, 0, 150] ],
        ["g", [0, 150, 0] ],
        ["b", [150, 0, 0] ],
        ["c", [150, 150, 0] ],
        ["m", [150, 0, 150] ],
        ["y", [0, 150, 150] ],
        ["w", [150, 150, 150]]
    ]

