# coding: utf-8
'''
Copyright (c) University of Strasbourg, All Rights Reserved.
'''
import numpy as np
import cv2
import subprocess
import os
import random
from IPython.display import HTML, display
import colorsys

color_brightness = 150
cc = {}
cc["r"] = (0, 0, color_brightness)
cc["g"] = (0, color_brightness, 0)
cc["b"] = (color_brightness, 0, 0)
cc["c"] = (color_brightness, color_brightness, 0)
cc["m"] = (color_brightness, 0, color_brightness)
cc["y"] = (0, color_brightness, color_brightness)
cc["w"] = (color_brightness, color_brightness, color_brightness)
cc["k"] = (0, 0, 0)
cc["t1"] = (205, 97, 85)
cc["t2"] = (33, 97, 140)
cc["t3"] = (23, 165, 137)
cc["t4"] = (125, 102, 8)
cc["t5"] = (230, 126, 34)
cc["t6"] = (211, 84, 0)
cc["t7"] = (52, 73, 94)
cc["t8"] = (102, 255, 153)
cc["t9"] = (51, 0, 204)
cc["t10"] = (255, 0, 204)
cc["t11"] = (230, 126, 34)
cc["t12"] = (211, 84, 0)
cc["t13"] = (52, 73, 94)
cc["t14"] = (102, 255, 153)
cc["t15"] = (51, 0, 204)
cc["t16"] = (255, 0, 204)
cc["t17"] = (255, 0, 204)
cc["t18"] = (66, 204, 150)

cir_radius = 3
max_persons = 100

""" predefined first 20 colors """
colors_arr = {}
colors_arr[0] = (0, 0, color_brightness)
colors_arr[1] = (0, color_brightness, 0)
colors_arr[2] = (color_brightness, 0, 0)
colors_arr[3] = (color_brightness, color_brightness, 0)
colors_arr[4] = (color_brightness, 0, color_brightness)
colors_arr[5] = (0, color_brightness, color_brightness)
colors_arr[6] = (color_brightness, color_brightness / 2, 255 - color_brightness / 2)
colors_arr[7] = (color_brightness / 2, 255 - color_brightness / 2, color_brightness)
colors_arr[8] = (color_brightness, color_brightness / 2, 255 - color_brightness / 2)
colors_arr[9] = (color_brightness / 2, 255 - color_brightness / 2, color_brightness / 2)
colors_arr[10] = (0, 0, color_brightness / 2)
colors_arr[11] = (0, color_brightness / 2, 0)
colors_arr[12] = (color_brightness / 2, 0, 0)
colors_arr[13] = (color_brightness / 2, color_brightness / 2, 0)
colors_arr[14] = (color_brightness / 2, 0, color_brightness / 2)
colors_arr[15] = (0, color_brightness / 2, color_brightness / 2)
colors_arr[16] = (color_brightness / 2, color_brightness / 4, 255 - color_brightness / 4)
colors_arr[17] = (color_brightness / 4, 255 - color_brightness / 4, color_brightness / 2)
colors_arr[18] = (color_brightness / 2, color_brightness / 4, 255 - color_brightness / 4)
colors_arr[19] = (color_brightness / 4, 255 - color_brightness / 4, color_brightness / 4)
camma_colors = ["t1", "t1", "t1", "t1", "t1", "t1", "t1", "t1", "t1", "t1"]
camma_colors_float = [
    (0.7, 0.7, 0.0, 0.5),
    (0.0, 0.7, 0.0, 0.5),
    (0.7, 0.0, 0.7, 0.5),
    (0.0, 0.7, 0.0, 0.5),
    (0.7, 0.0, 0.7, 0.5),
    (0.0, 0.7, 0.0, 0.5),
    (0.7, 0.0, 0.7, 0.5),
    (0.0, 0.7, 0.0, 0.5),
    (0.7, 0.0, 0.7, 0.5),
    (0.0, 0.7, 0.0, 0.5),
]
camma_pairs = [[0, 1], [1, 3], [3, 7], [3, 5], [7, 9], [1, 2], [2, 4], [2, 6], [6, 8], [4, 5]]
camma_colors_skeleton = ["y", "g", "g", "g", "g", "m", "m", "m", "m", "m"]

camma_part_names = [
    "head",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


coco_colors = [
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
    "t18",
]
coco_part_names = [
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
]
coco_pairs = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]
coco_colors_skeleton = [
    "m",
    "m",
    "g",
    "g",
    "r",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "m",
    "g",
]
h36_colors_skeleton = [
    "r",
    "r",
    "r",
    "r",
    "g",
    "g",
    "g",
    "m",
    "m",
    "m",
    "g",
    "g",
    "g",
    "m",
    "m",
    "m",
]

h36_pairs = [
    [9, 10],
    [10, 8],
    [8, 7],
    [7, 0],
    [0, 1],
    [1, 2],
    [2, 3],
    [0, 4],
    [4, 5],
    [5, 6],
    [8, 14],
    [14, 15],
    [15, 16],
    [8, 11],
    [11, 12],
    [12, 13],
]


def bgr2rgb(im):
    b, g, r = cv2.split(im)
    return cv2.merge([r, g, b])


def plt_imshow(im, ax):
    im = bgr2rgb(im)
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])
    # plt.subplots_adjust(left=0.01, bottom=0.01, right=1.0, top=1.0, wspace=0.01, hspace=0.01)
    ax.imshow(im)


def progress_bar(value, max=100):
    """ A HTML helper function to display the progress bar
    Args:
        value ([int]): [current progress bar value]
        max (int, optional): [maximum value]. Defaults to 100.
    Returns:
        [str]: [HTML progress bar string]
    """
    return HTML(
        """
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(
            value=value, max=max
        )
    )


def generate_random_colors(NUM=100):
    # generate random colors
    colors = []
    for _ in range(NUM):
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        colors.append([int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)])
    return colors


def coco_to_camma_kps(coco_kps):
    """
    convert coco keypoints(17) to camma-keypoints(10)
    :param coco_kps: 17 keypoints of coco
    :return: camma_keypoints
    """
    num_keypoints = 10
    camma_kps = np.zeros((num_keypoints, 3))

    nose = coco_kps[0, :]
    leye = coco_kps[1, :]
    reye = coco_kps[2, :]
    lear = coco_kps[3, :]
    rear = coco_kps[4, :]

    if leye[-1] > 0 and reye[-1] > 0:
        camma_kps[0, :] = ((leye + reye) / 2).reshape(1, 3)
    elif lear[-1] > 0 and rear[-1] > 0:
        camma_kps[0, :] = ((lear + rear) / 2).reshape(1, 3)
    elif nose[-1] > 0:
        camma_kps[0, :] = nose.reshape(1, 3)
    elif reye[-1] > 0:
        camma_kps[0, :] = reye.reshape(1, 3)
    elif leye[-1] > 0:
        camma_kps[0, :] = leye.reshape(1, 3)
    elif rear[-1] > 0:
        camma_kps[0, :] = rear.reshape(1, 3)
    elif lear[-1] > 0:
        camma_kps[0, :] = lear.reshape(1, 3)
    else:
        camma_kps[0, :] = np.array([0, 0, 0]).reshape(1, 3)

    lshoulder = coco_kps[5, :]
    rshoulder = coco_kps[6, :]
    lhip = coco_kps[11, :]
    rhip = coco_kps[12, :]
    lelbow = coco_kps[7, :]
    relbow = coco_kps[8, :]
    lwrist = coco_kps[9, :]
    rwrist = coco_kps[10, :]

    if lshoulder[-1] > 0 and rshoulder[-1] > 0:
        camma_kps[1, :] = ((lshoulder + rshoulder) / 2).reshape(1, 3)
    elif rshoulder[-1] > 0:
        camma_kps[1, :] = rshoulder.reshape(1, 3)
    elif lshoulder[-1] > 0:
        camma_kps[1, :] = lshoulder.reshape(1, 3)
    else:
        camma_kps[1, :] = np.array([0, 0, 0]).reshape(1, 3)

    if lshoulder[-1] > 0:
        camma_kps[2, :] = lshoulder.reshape(1, 3)

    if rshoulder[-1] > 0:
        camma_kps[3, :] = rshoulder.reshape(1, 3)

    if lhip[-1] > 0:
        camma_kps[4, :] = lhip.reshape(1, 3)

    if rhip[-1] > 0:
        camma_kps[5, :] = rhip.reshape(1, 3)

    if lelbow[-1] > 0:
        camma_kps[6, :] = lelbow.reshape(1, 3)

    if relbow[-1] > 0:
        camma_kps[7, :] = relbow.reshape(1, 3)

    if lwrist[-1] > 0:
        camma_kps[8, :] = lwrist.reshape(1, 3)

    if rwrist[-1] > 0:
        camma_kps[9, :] = rwrist.reshape(1, 3)

    return camma_kps


def images_to_video(img_folder, output_vid_file, fps=20):
    """[convert png images to video using ffmpeg]
    Args:
        img_folder ([str]): [path to images]
        output_vid_file ([str]): [Name of the output video file name]
    """
    os.makedirs(img_folder, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-threads",
        "16",
        "-i",
        f"{img_folder}/%06d.png",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-v",
        "error",
        output_vid_file,
    ]
    print(f'\nRunning "{" ".join(command)}"')
    subprocess.call(command)
    print("\nVideo generation finished")


def coco_to_h36_kps(pose):
    # 0 represent not match
    valid_match = [0, 12, 14, 16, 11, 13, 15, 0, 0, 0, 0, 5, 7, 9, 6, 8, 10]
    _pose = pose[valid_match]
    # pelvis is the mid point of lhip and rhip
    if pose[11][-1] > 0 and pose[12][-1] > 0:
        _pose[0] = (pose[11] + pose[12]) / 2.0
    elif pose[11][-1] > 0:
        _pose[0] = pose[11]
    elif pose[12][-1] > 0:
        _pose[0] = pose[12]
    else:
        _pose[0] = np.array([0, 0, 0]).reshape(1, 3)
    # neck is the mid point of shoulder
    if pose[5][-1] > 0 and pose[6][-1] > 0:
        _pose[8] = (pose[5] + pose[6]) / 2.0
    elif pose[5][-1] > 0:
        _pose[8] = pose[5]
    elif pose[6][-1] > 0:
        _pose[8] = pose[6]
    else:
        _pose[8] = np.array([0, 0, 0]).reshape(1, 3)

    # head top are the mid points of ears
    if pose[3][-1] > 0 and pose[4][-1] > 0:
        _pose[9] = (pose[3] + pose[4]) / 2.0
    elif pose[3][-1] > 0:
        _pose[9] = pose[3]
    elif pose[4][-1] > 0:
        _pose[9] = pose[4]
    else:
        _pose[9] = np.array([0, 0, 0]).reshape(1, 3)
    _pose[10] = _pose[9]

    # throax is the mid point of neck and pelvis
    if _pose[0][-1] > 0 and _pose[8][-1] > 0:
        _pose[7] = (_pose[0] + _pose[8]) / 2.0
    elif _pose[0][-1] > 0:
        _pose[7] = _pose[0]
    elif _pose[8][-1] > 0:
        _pose[7] = _pose[8]
    else:
        _pose[7] = np.array([0, 0, 0]).reshape(1, 3)

    return _pose


def rect_prism(ax, x_range, y_range, z_range):
    """
    plot the 3d rectangle
    :param x_range:
    :param y_range:
    :param z_range:
    :return:
    """
    # xx, yy = np.meshgrid(x_range, y_range)
    # zz = z_range[0] * np.ones(xx.shape)
    # ax.plot_wireframe(xx, yy, zz, color="darkgray")
    # ax.plot_surface(xx, yy, zz, color="darkgray")

    # zz = z_range[1] * np.ones(xx.shape)
    # ax.plot_wireframe(xx, yy, zz, color="darkgray")
    # ax.plot_surface(xx, yy, zz, color="darkgray")

    # yy, zz = np.meshgrid(y_range, z_range)
    # xx = x_range[0] * np.ones(yy.shape)
    # ax.plot_wireframe(xx, yy, zz, color="darkgray")
    # ax.plot_surface(xx, yy, zz, color="darkgray")

    # xx = x_range[1] * np.ones(yy.shape)
    # ax.plot_wireframe(xx, yy, zz, color="darkgray")
    # ax.plot_surface(xx, yy, zz, color="darkgray")

    # xx, zz = np.meshgrid(x_range, z_range)
    # yy = y_range[0] * np.ones(zz.shape)
    # ax.plot_wireframe(xx, yy, zz, color="darkgray")
    # ax.plot_surface(xx, yy, zz, color="darkgray")

    # yy = y_range[1] * np.ones(zz.shape)
    # ax.plot_wireframe(xx, yy, zz, color="darkgray")
    # ax.plot_surface(xx, yy, zz, color="darkgray")

    # plot the surface on the bottom
    xx, zz = np.meshgrid([-100, 840], [2, 10])
    yy = 520 * np.ones(xx.shape)
    ax.plot_wireframe(xx, yy, zz, color="dimgray", linestyle="dotted", alpha=1.0)
    ax.plot_surface(xx, yy, zz, color="dimgray", alpha=0.6)
    # xx, yy = np.meshgrid([0, 640], [470, 480])
    # zz = 10 * np.ones(xx.shape)
    # ax.plot_wireframe(xx, yy, zz, color="b")
    # ax.plot_surface(xx, yy, zz, color="b", alpha=0.9)


def bgr2rgb(im):
    """[convert opencv image in BGR format to RGB format]
    Args:
        im ([numpy.ndarray]): [input image in BGR format]
    Returns:
        [numpy.ndarray]: [output image in RGB format]
    """
    b, g, r = cv2.split(im)
    return cv2.merge([r, g, b])


def draw_2d_keypoints(image, pt2d, style="camma", box=None, use_color=None, THRESH=0.05):
    """
    pt2d can be 17,3 for h36 or coco and 10,3 for camma
    """
    if style == "camma":
        colors_skeleton = camma_colors_skeleton
        pairs = camma_pairs
    elif style == "coco":
        colors_skeleton = coco_colors_skeleton
        pairs = coco_pairs
    elif style == "h36":
        colors_skeleton = h36_colors_skeleton
        pairs = h36_pairs

    # draw keypoints
    for idx in range(len(colors_skeleton)):
        if use_color is None:
            color = cc[colors_skeleton[idx]]
        else:
            color = use_color
        pair = pairs[idx]
        pt1, sc1 = tuple(pt2d[pair[0], :].astype(int)[0:2]), pt2d[pair[0]][2]
        pt2, sc2 = tuple(pt2d[pair[1], :].astype(int)[0:2]), pt2d[pair[1]][2]
        if sc1 > THRESH and sc2 > THRESH:
            if 0 not in pt1 + pt2:
                cv2.line(image, pt1, pt2, color, 2, cv2.LINE_AA)
                cv2.circle(image, pt1, 2, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(image, pt2, 2, (0, 0, 0), 3, cv2.LINE_AA)

    # draw box
    if box is not None:
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            use_color if use_color else (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )


def cam_to_image_coordinate(pts, cam):
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    pts[:, 0] = pts[:, 0] / pts[:, 2]
    pts[:, 1] = pts[:, 1] / pts[:, 2]
    pts[:, 0] = pts[:, 0] * fx + cx
    pts[:, 1] = pts[:, 1] * fy + cy
    pts[:, 2] = pts[:, 2] / 1000.0
    return pts
