import cv2
import matplotlib.pyplot as plt
import numpy as np
from calibrate_camera import get_calibration_data, undistort_image


LINE_THICKNESS = 4
LINE_COLOR = [255, 0, 0]
TOP = 450
BOTTOM = 720

# Source points:
left_top = (592, TOP)
left_bottom = (180, BOTTOM)
right_top = (686, TOP)
right_bottom = (1120, BOTTOM)

src = np.float32([
    [left_top[0], left_top[1]],
    [right_top[0], right_top[1]],
    [right_bottom[0], right_bottom[1]],
    [left_bottom[0], left_bottom[1]]
])

dst_left = 320
dst_right = 960
dst = np.float32([
    [dst_left, 0],
    [dst_right, 0],
    [dst_right, 720],
    [dst_left, 720]
])


def warper(image, src, dst):
    img_size = (img.shape[1], img.shape[0])

    perspective_transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, perspective_transform, img_size)

    return warped


if __name__ == '__main__':
    calibration = get_calibration_data()
    matrix = calibration['mtx']
    distortion_coefficients = calibration['dist']

    orig_img = cv2.cvtColor(cv2.imread('./test_images/straight_lines1.jpg'), cv2.COLOR_BGR2RGB)
    img = undistort_image(orig_img, matrix, distortion_coefficients)
    img_size = (img.shape[1], img.shape[0])

    img_copy = np.copy(img)
    cv2.line(img_copy, left_bottom, left_top, LINE_COLOR, LINE_THICKNESS)
    cv2.line(img_copy, right_bottom, right_top, LINE_COLOR, LINE_THICKNESS)
    cv2.line(img_copy, left_top, right_top, LINE_COLOR, LINE_THICKNESS)
    cv2.line(img_copy, left_bottom, right_bottom, LINE_COLOR, LINE_THICKNESS)

    # matrix = cv2.getPerspectiveTransform(src, dst)
    warped = warper(img, src, dst)
    warped_lines = np.copy(warped)
    cv2.line(warped_lines, (dst_left, 0), (dst_left, 720), LINE_COLOR, LINE_THICKNESS)
    cv2.line(warped_lines, (dst_right, 0), (dst_right, 720), LINE_COLOR, LINE_THICKNESS)

    plt.figure(figsize=(24, 15))
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # plt.tight_layout()
    plt.subplot(321)
    plt.title('Original Image', fontsize=24)
    plt.imshow(orig_img)

    plt.subplot(322)
    plt.title('Undistorted', fontsize=24)
    plt.imshow(img)
    # ax3.imshow(threshed, cmap='gray')
    # ax3.set_title('Thresholded Gradient [Sobel]', fontsize=24)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.subplot(323)
    plt.title('Region of Interest', fontsize=24)
    plt.imshow(img_copy)

    plt.subplot(324)
    plt.title('Warped', fontsize=24)
    plt.imshow(warped_lines);
