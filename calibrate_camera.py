import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

# %matplotlib qt
# %matplotlib inline


class ImageSample():

    def __init__(self, filename):
        self.filename = filename
        self.image = self.read_image(self.filename)
        self.size = (self.image.shape[1], self.image.shape[0])

    def read_image(self, filename):
        return cv2.imread(filename)


class Calibrator():

    def __init__(self, calibration):
        self.ret = calibration[0]
        self.mtx = calibration[1]
        self.dist = calibration[2]
        self.rvecs = calibration[3]
        self.tvecs = calibration[4]


def get_calibration_points(images_regex, num_channels, x, y, z=0):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((y * x, num_channels), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    # images = glob.glob('calibration_wide/GO*.jpg')
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
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


def get_undistorted_img(filename, calibration):
    img = cv2.imread(filename)
    img_size = (img.shape[1], img.shape[0])
    dst = cv2.undistort(img, calibration.mtx, calibration.dist, None, calibration.mtx)
    return dst


objpoints, imgpoints = get_calibration_points('./camera_cal/calibration*.jpg', 3, 9, 6, 0)

img_sample = ImageSample('./camera_cal/calibration1.jpg')

# Calibrate camera
calibration = calibrate_camera(objpoints, imgpoints, img_sample.size, './output_images/wide_dist_pickle.p')

# Test undistortion on an image
dst = get_undistorted_img(img_sample.filename, calibration)
cv2.imwrite('./output_images/test_undist.jpg', dst)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img_sample.image)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)