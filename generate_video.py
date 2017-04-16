import cv2
import numpy as np
from calibrate_camera import get_calibration_data, undistort_image
from moviepy.editor import VideoFileClip
from thresholds import get_color_threshold
from warper import get_dst_pts, get_src_pts, warper

calibration = get_calibration_data()
matrix = calibration['mtx']
distortion_coefficients = calibration['dist']

frames = []

def process_window(
    binary_warped,
    window,
    window_height,
    leftx_current,
    rightx_current,
    margin,
    nonzeroy,
    nonzerox,
    left_lane_inds,
    right_lane_inds,
    minpix
):
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

    # If found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    return left_lane_inds, right_lane_inds, leftx_current, rightx_current


def add_text_to_final_image(
    image,
    labels_with_pos,
    color,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    fontscale=1.25,
    thickness=2,
    lineType=cv2.LINE_AA,
    drop_shadow_offset=3
):
    drop_shadow_color = (0, 0, 0)

    for label in labels_with_pos:
        text = label[0]
        pos = label[1]
        pos_offset = (pos[0] + drop_shadow_offset, pos[1] + drop_shadow_offset)

        # Draw drop shadow
        cv2.putText(
            img=image,
            text=text,
            org=pos_offset,
            fontFace=font,
            fontScale=fontscale,
            color=drop_shadow_color,
            thickness=thickness,
            lineType=lineType
        )

        # Draw main color
        cv2.putText(
            img=image,
            text=text,
            org=pos,
            fontFace=font,
            fontScale=fontscale,
            color=color,
            thickness=thickness,
            lineType=lineType
        )
    return image


def pipeline(image):
    """Pipeline to manipulate an image to detect lane lines and illustrate the lane region
    
    :param numpy.ndarray image:  Image containing lanes to be detected
    :return numpy.ndarray: Image with lane painted with curvature radius and distance-from-center labels
    """
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
    margin = 155

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        left_lane_inds, right_lane_inds, leftx_current, rightx_current = process_window(
            binary_warped,
            window,
            window_height,
            leftx_current,
            rightx_current,
            margin,
            nonzeroy,
            nonzerox,
            left_lane_inds,
            right_lane_inds,
            minpix
        )

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
    # Let's choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

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

    lane_center = (leftx_base + rightx_base) / 2
    lane_off_center = (lane_center - (image.shape[1] / 2)) * xm_per_pix
    text_center_off = 'Vehicle is {:.4f} m {} of center'.format(abs(lane_off_center), 'left' if lane_off_center > 0 else 'right')

    # Draw labels
    color = (255, 255, 0)
    labels_with_pos = [
        (text_curve_radius, (50, 100)),
        (text_center_off, (50, 160))
    ]

    undistorted = add_text_to_final_image(
        undistorted,
        labels_with_pos,
        color
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

    # video_output = 'fix.mp4'
    # clip1 = VideoFileClip("project_video.mp4").subclip(39, 40)
    # yellow_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    # yellow_clip.write_videofile(video_output, audio=False)
