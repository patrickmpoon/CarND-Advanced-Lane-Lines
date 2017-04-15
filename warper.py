import cv2
import numpy as np


SRC_TOP = 450
SRC_BOTTOM = 720
SRC_LEFT_TOP_X = 592
SRC_LEFT_BOTTOM_X = 180
SRC_RIGHT_TOP_X = 686
SRC_RIGHT_BOTTOM_X = 1120

DST_LEFT = 320
DST_RIGHT = 960
IMAGE_HEIGHT = 720


def warper(image, src, dst):
    img_size = (image.shape[1], image.shape[0])

    perspective_transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, perspective_transform, img_size)

    return warped


def get_src_pts(
    top=SRC_TOP,
    bottom=SRC_BOTTOM,
    left_top_x=SRC_LEFT_TOP_X,
    left_bottom_x=SRC_LEFT_BOTTOM_X,
    right_top_x=SRC_RIGHT_TOP_X,
    right_bottom_x=SRC_RIGHT_BOTTOM_X
):
    return np.float32([
        [left_top_x, top],
        [right_top_x, top],
        [right_bottom_x, bottom],
        [left_bottom_x, bottom]
    ])


def get_dst_pts(left=DST_LEFT, right=DST_RIGHT, top=0, bottom=IMAGE_HEIGHT):
    return np.float32([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ])
