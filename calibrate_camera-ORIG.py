import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

# %matplotlib qt
# %matplotlib inline


class Image():

    def __init__(self, filename):
        self.filename = filename
        self.image = self.read_image(self.filename)
        self.size = (self.image.shape[0], self.image.shape[1])
        self.gray = self.to_gray(self.image)

    def read_image(self, filename):
        return cv2.imread(filename)

    def to_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class Calibrator():

    def __init__(self, calibration):
        self.ret = calibration[0]
        self.mtx = calibration[1]
        self.dist = calibration[2]
        self.rvecs = calibration[3]
        self.tvecs = calibration[4]


def get_corners(image, x, y, flags=None):
    corners = cv2.findChessboardCorners(image, (x, y), flags)
    return corners


def get_calibration_points(images_regex, num_channels, x, y, z=0):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((y * x, num_channels), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(images_regex)

    # Step through the list and search for chessboard corners
    for idx, filename in enumerate(images):
        img = Image(filename)

        # Find the chessboard corners
        has_found_corners, corners = get_corners(img.image, x, y)

        # If found, add object points, image points
        if has_found_corners == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
#             cv2.drawChessboardCorners(img.image, (9,6), corners, has_found_corners)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)

            # cv2.imshow('img', img)

            # cv2.waitKey(500)

    return objpoints, imgpoints


def calibrate_camera(objpoints, imgpoints, img_size, output_file):
    calibration = Calibrator(cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None))

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = calibration.mtx
    dist_pickle["dist"] = calibration.dist
    pickle.dump(dist_pickle, open(output_file, "wb"))

    return calibration


def undistort_img(img, calibration):
    undistorted = cv2.undistort(img, calibration.mtx, calibration.dist, None, calibration.mtx)
    return undistorted


def get_src_pts(corners, x):
    return np.float32([corners[0], corners[x-1], corners[-1], corners[-x]])


def get_dst_pts(offset, img_size):
    return np.float32([[offset, offset], [img_size[0]-offset, offset],
                         [img_size[0]-offset, img_size[1]-offset],
                         [offset, img_size[1]-offset]])


def get_matrix(src_pts, dst_pts):
    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def get_warped(undist_img, matrix, img_size):
    return cv2.warpPerspective(undist_img, matrix, img_size)


num_channels = 3
x = 9
y = 6
z = 0
objpoints, imgpoints = get_calibration_points('./camera_cal/calibration*.jpg', num_channels, x, y, z)

src = Image('./camera_cal/calibration2.jpg')

# Calibrate camera
calibration = calibrate_camera(objpoints, imgpoints, src.image.shape[::-1], './output_images/calibration.p')

# Test undistortion on an image
dst = undistort_img(src.image, calibration)
# dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

has_found_corners, corners = get_corners(dst, x, y)
cv2.drawChessboardCorners(dst, (x, y), corners, has_found_corners)
plt.figure(figsize=(35,14))
plt.subplot(121)
plt.axis('off')
plt.title('Drawn')
plt.imshow(dst)

# cv2.imwrite('./output_images/test_undist.jpg', dst)

# dst_offset = 100
# src_pts = get_src_pts(corners, x)
# dst_pts = get_dst_pts(dst_offset, src.size)
# matrix = get_matrix(src_pts, dst_pts)
# warped = get_warped(dst, matrix, src.size)

# # Visualize undistortion
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
# ax1.imshow(src.image)
# ax1.set_title('Original Image', fontsize=18)
# ax2.imshow(dst)
# ax2.set_title('Undistorted Image', fontsize=18)
# ax3.imshow(warped)
# ax3.set_title('Undistorted and Warped Image', fontsize=18);
