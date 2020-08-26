# coding: utf-8
'''
Copyright (c) University of Strasbourg, All Rights Reserved.
'''
import numpy as np
np.set_printoptions(precision=2)
import json
import logging

logger = logging.getLogger("detectron2")


class MPJPE_Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.total_error = np.zeros(10)
        self.count = 0
        self.results_3d = []
        with open(cfg.DATASETS.H36_STATS_PATH) as f:
            h36stats = json.load(f)
            mu2d = np.array(h36stats["data_mean_2d"])[h36stats["dim_to_use_2d"]]
            std2d = np.array(h36stats["data_std_2d"])[h36stats["dim_to_use_2d"]]
            mu3d = np.array(h36stats["data_mean_3d"])
            std3d = np.array(h36stats["data_std_3d"])
            dimuse3d = np.array(h36stats["dim_to_use_3d"])
            self.h36stats = {
                "mu2d": mu2d,
                "std2d": std2d,
                "mu3d": mu3d,
                "std3d": std3d,
                "dimuse3d": dimuse3d,
            }

    def unNormalizeData(self, normalized_data, data_mean, data_std, dimensions_to_use):
        T = normalized_data.shape[0]  # Batch size
        D = data_mean.shape[0]  # 96

        orig_data = np.zeros((T, D), dtype=np.float32)

        orig_data[:, dimensions_to_use] = normalized_data

        # Multiply times stdev and add the mean
        stdMat = data_std.reshape((1, D))
        stdMat = np.repeat(stdMat, T, axis=0)
        meanMat = data_mean.reshape((1, D))
        meanMat = np.repeat(meanMat, T, axis=0)
        orig_data = np.multiply(orig_data, stdMat) + meanMat
        return orig_data

    def H36toCAMMA(self, pose):
        index_loc = [9, 8, 11, 14, 4, 1, 12, 15, 13, 16]
        return np.copy(pose[index_loc])

    def process(self, inputs, outputs):
        # get ground truths having person_id >= 0
        if (
            len(inputs[0]["instances"]) == 0
            or (inputs[0]["instances"].person_ids_camma == -1).all()
        ):
            return

        # init
        coco_sigmas = (
            np.array(
                [
                    0.26,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.79,
                    0.79,
                    0.72,
                    0.72,
                    0.62,
                    0.62,
                    1.07,
                    1.07,
                    0.87,
                    0.87,
                    0.89,
                    0.89,
                ]
            )
            / 10.0
        )
        vars = (coco_sigmas * 2) ** 2
        k = len(coco_sigmas)

        # get the 2D boxes, 2D keypoints and 3D keypoints for which person_id is valid
        valid = inputs[0]["instances"].person_ids_camma != -1
        image_id = inputs[0]["image_id"]
        gt_kps = inputs[0]["instances"].gt_keypoints.tensor[valid].numpy()
        gt_boxes = inputs[0]["instances"].gt_boxes.tensor[valid].numpy()
        gt_kps3d = inputs[0]["instances"].anns3d_camma_allkps[valid].numpy()
        gt_kps2d_camma = inputs[0]["instances"].anns2d_camma[valid].numpy()
        focal_length = inputs[0]["instances"].focal_length.numpy()
        principal_point = inputs[0]["instances"].principal_point.numpy()
        fx_d, fy_d = focal_length[0], focal_length[1]
        cx_d, cy_d = principal_point[0], principal_point[1]

        # get the 2D keypoint detections and 2D depth detections
        ious = np.zeros((gt_kps.shape[0], outputs[0]["instances"].scores.shape[0]))

        # compute the ious
        # taken from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        for j, gt in enumerate(gt_kps):
            xg = gt[:, 0]
            yg = gt[:, 1]
            vg = gt[:, 2]
            k1 = np.count_nonzero(vg > 0)
            bb = gt_boxes[j]
            bb[2] = bb[2] - bb[0]
            bb[3] = bb[3] - bb[1]
            area = bb[2] * bb[3]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(outputs[0]["instances"].pred_keypoints):
                dt = dt.cpu().numpy()
                xd = dt[:, 0]
                yd = dt[:, 1]
                if k1 > 0:
                    dx = xd - xg
                    dy = yd - yg
                else:
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / (area + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[j, i] = np.sum(np.exp(-e)) / e.shape[0]
        #
        # get the matched detections
        if not ious.size == 0:
            dets_ids = np.argmax(ious, axis=1)
            preds_3d = outputs[0]["instances"].pred_keypoints_h36_3d[dets_ids].cpu().numpy()
            preds_2d_h36 = outputs[0]["instances"].pred_keypoints_h36_2d[dets_ids].cpu().numpy()
            preds_2d_coco = outputs[0]["instances"].pred_keypoints[dets_ids].cpu().numpy()
            assert gt_kps.shape[0] == preds_3d.shape[0]

            # loop to compute the mpjpe
            for i, gt_3d in enumerate(gt_kps3d):
                gt_2d = gt_kps[i]
                pose3d = preds_3d[i]
                dt_2d_h36 = preds_2d_h36[i]
                dt_2d_coco = preds_2d_coco[i]
                gt_3d *= 1000  # convert meter to mm
                pelvis_gt = (gt_3d[4, 0:3] + gt_3d[5, 0:3]) / 2.0
                gt_3d_rel = gt_3d[:, 0:3] - pelvis_gt
                visible = (gt_3d[:, 3] > 0).astype(np.float32)
                gt_3d_rel = gt_3d_rel[:, 0:3] * visible[:, np.newaxis]

                pose3d = self.unNormalizeData(
                    pose3d, self.h36stats["mu3d"], self.h36stats["std3d"], self.h36stats["dimuse3d"]
                )
                dim_use = np.hstack((np.arange(3), self.h36stats["dimuse3d"]))
                pose3d = pose3d[:, dim_use][0]
                pose3d_camma = self.H36toCAMMA(np.array(pose3d).reshape(17, 3))
                pose3d_camma = pose3d_camma[:, 0:3] * visible[:, np.newaxis]

                error = np.sqrt(np.sum((pose3d_camma - gt_3d_rel) ** 2, axis=1))
                self.total_error += error
                self.count += 1
                self.results_3d.append(
                    {
                        "image_id": image_id,
                        "pelvis_gt": pelvis_gt,
                        "gt_2d": gt_2d,
                        "gt_3d": gt_3d,
                        "gt_3d_rel": gt_3d_rel,
                        "dt_2d_h36": dt_2d_h36,
                        "dt_2d_coco": dt_2d_coco,
                        "dt_3d": pose3d,
                        "dt_3d_camma": pose3d_camma,
                        "error": error,
                    }
                )
                
    def evaluate(self):
        self.total_error /= self.count
        head = self.total_error[0]
        neck = self.total_error[1]
        shoulder = (self.total_error[2] + self.total_error[3]) / 2.0
        hip = (self.total_error[4] + self.total_error[5]) / 2.0
        elbow = (self.total_error[6] + self.total_error[7]) / 2.0
        wrist = (self.total_error[8] + self.total_error[9]) / 2.0
        logger.info(
            "total error for each joint: head, neck, l_shoud, r_should, l_hip, r_hip, l_elbow, r_elbow, l_wrist, r_wrist"
        )
        logger.info("total error for each joint: {} mm".format(self.total_error))

        logger.info(
            "total error selected joint in mm : shoulder={}, hip={}, elbow={}, wrist={}, avg={}".format(
                shoulder, hip, elbow, wrist, ((shoulder + hip + elbow + wrist) / 4.0)
            )
        )
        logger.info(
            "total error all joint in mm : head={}, neck={}, shoulder={}, hip={}, elbow={}, wrist={}, avg={}".format(
                head, neck, shoulder, hip, elbow, wrist, ((head+neck+2*shoulder+2*hip+2*elbow+2*wrist)/10.0)
            )
        )
        logger.info("total error selected joints sanity check: {} mm".format(np.sum(self.total_error[2:]) / 8))
        return self.results_3d

