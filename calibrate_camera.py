import cv2
from glob import iglob
import matplotlib.pyplot as plt
import numpy as np
import pickle


def get_corners(gray_image, pts_per_row, pts_per_column, flags=None):
    return cv2.findChessboardCorners(gray_image, (pts_per_row, pts_per_column), flags)


def draw_image_corners(image, pts_per_row, pts_per_column, corners, ret):
    cornered = cv2.drawChessboardCorners(image, (pts_per_row, pts_per_column), corners, ret)
    return cornered


def get_calibration_pts(img_files_regex, pts_per_row, pts_per_column, num_channels):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pts_per_column * pts_per_row, 3), np.float32)
    objp[:,:2] = np.mgrid[0:pts_per_row, 0:pts_per_column].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = iglob(img_files_regex)

    # Step through the list and search for chessboard corners
    for idx, filename in enumerate(images, start=1):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = get_corners(gray, pts_per_row, pts_per_column)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = draw_image_corners(img, pts_per_row, pts_per_column, corners, ret)
            write_name = './test_out/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)

    return objpoints, imgpoints


def undistort_image(image, matrix, distortion_coefficients):
    return cv2.undistort(image, matrix, distortion_coefficients, None, matrix)


if __name__ == '__main__':
    img_files_regex = './camera_cal/calibration*.jpg'
    pts_per_row = 9
    pts_per_column = 6
    num_channels = 3

    objpoints, imgpoints = get_calibration_pts(img_files_regex, pts_per_row, pts_per_column, num_channels)

    # Test undistortion on an image
    img = cv2.imread('./camera_cal/calibration3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration results
    dist_pickle = {}
    dist_pickle["mtx"] = matrix
    dist_pickle["dist"] = distortion_coefficients
    pickle.dump( dist_pickle, open( "./output_images/calibration.p", "wb" ) )

    # Undistort image and save results to file
    undistorted = undistort_image(img, matrix, distortion_coefficients)
    undistorted_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    ret, corners = get_corners(undistorted_gray, pts_per_row, pts_per_column)
    undistorted = draw_image_corners(undistorted, pts_per_row, pts_per_column, corners, ret)
    cv2.imwrite('test_out/test_undist.jpg', undistorted)

    # Visualize distortion correction
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=30)
