import cv2
import numpy as np
from calibrate_camera import get_calibration_data, undistort_image
from moviepy.editor import VideoFileClip
from thresholds import get_color_threshold
from warper import get_dst_pts, get_src_pts, warper

calibration = get_calibration_data()
matrix = calibration['mtx']
distortion_coefficients = calibration['dist']


def pipeline(image):
    undistorted = undistort_image(image, matrix, distortion_coefficients)
    _, combined_binary = get_color_threshold(undistorted)

    src = get_src_pts()
    dst = get_dst_pts()
    binary_warped = warper(combined_binary, src, dst)

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    print('midpoint = {}'.format(midpoint))
    print('leftx_base = {}  rightx_base = {}'.format(leftx_base, rightx_base))

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - ((window + 1) * window_height)
        win_y_high = binary_warped.shape[0] - (window * window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (255, 255, 0))

    avg_curve_radius = np.mean([left_curverad + right_curverad])
    text_curve_radius = 'Curvature Radius = {:.4f} (m)'.format(avg_curve_radius)

    lane_center = (leftx_current + rightx_current) / 2
    lane_off_center = (lane_center - (image.shape[1] / 2)) * xm_per_pix
    text_center_off = 'Vehicle is {:.4f} m {} of center'.format(abs(lane_off_center), 'left' if lane_off_center > 0 else 'right')

    # Drop shadow
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1.25
    thickness = 2
    cv2.putText(
        undistorted,
        text_curve_radius,
        (53, 103),
        font,
        fontscale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )
    cv2.putText(
        undistorted,
        text_curve_radius,
        (50, 100),
        font,
        fontscale,
        (255, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    # Primary color
    cv2.putText(
        undistorted,
        text_center_off,
        (53, 163),
        font,
        fontscale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )
    cv2.putText(
        undistorted,
        text_center_off,
        (50, 160),
        font,
        fontscale,
        (255, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warper(color_warp, dst, src)

    # Combine the result with the original image
    return cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)


if __name__ == "__main__":
    video_output = 'yellow_brick_road.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    yellow_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    yellow_clip.write_videofile(video_output, audio=False)