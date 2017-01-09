## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project is to use advanced technologies to solve lane detection problem.

### Image/Frame Processing:
The steps of this project are the following:  

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. The images for camera calibration are stored in the folder called `camera_cal`. Then I save the amera calibration matrix **[mtx]** and distortion coefficients **[dist]** to a pickle file `wide_dist_pickle.p` 
    * Find the chess board points which are (9x6)
    ![ChessBoard Corners](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/find_orners.png?raw=true)
    * Use 20 images and cv2 functions to calculate camera calibration matrix and distortion coefficients
    > ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    ![ChessBoard Distortion](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/carlibre_chessboard.png?raw=true)
    * Apply matrix and coefficients on Test images result
    > test_1![test_1](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/test1_calibre_calibre.png?raw=true)
    > test_2![test_2](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/test2_calibre.png?raw=true)
    > test_3![test_3](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/test3_calibre.png?raw=true)
    > test_4![test_4](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/test4_calibre.png?raw=true)
    > test_5![test_5](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/test5_calibre.png?raw=true)
    > test_6![test_6](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/test6_calibre.png?raw=true)
    > solidWhiteRight![solidWhiteRight](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/solidWhiteRight_calibre.png?raw=true)
    > solidYellowLeft![solidYellowLeft](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/solidYellowLeft_calibre.png?raw=true)

2. Use color transforms, gradients, etc., to create a thresholded binary image.
	* Read in the camera calibration matrix and distortion coefficients
	* Undistort image using calibre matrix (Results shown above)
	* Using methods to get binary images
	 * X-axis Gratituds 
	 > abs_sobel_thresh(indist, orient='x', sobel_kernel=ksize, thresh=(20, 250))
	 > ![X-axis Gratituds ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/gradx.jpg?raw=true)
	 * Y-axis Gratituds 
	 > abs_sobel_thresh(indist, orient='y', sobel_kernel=ksize, thresh=(50, 250))
	 > ![Y-axis Gratituds ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/grady.jpg?raw=true)
	 * Gradient Magnitude thresh fllter 
	 > mag_binary = utils.mag_thresh(indist, sobel_kernel=ksize, mag_thresh=(40, 250))
	 > ![Gradient Magnitude ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/gradient_mag.jpg?raw=true)
	  * Gradient direction thresh fllter 
	 > dir_binary = utils.dir_threshold(indist, sobel_kernel=ksize, thresh=(np.pi/6, np.pi/2))
	 > ![Gradient direction ](https://github.com/shangliy/advanced_lane/blob/master/gradient_dir.jpg?raw=true)
	  * HLS S-channel thresh fllter
	 > s_binary = utils.hls_select(indist, thresh=(100, 255))
	 > ![HLS S-channel thresh fllter ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/diag_s.jpg?raw=true)
	  * **Combine filters together** **
	 > combined[ (s_binary == 255)|((((gradx == 1)) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
	 > ![Combine filters  ](https://github.com/shangliy/advanced_lane/blob/master/diag_com.jpg?raw=true)
	 
3. Apply a perspective transform to rectify binary image ("birds-eye view"). 
	Here I use masked area,canny edge and hougline to get the corresponding points. Then calculate the perspective transform matrix and inverse matrix.
    >  M = cv2.getPerspectiveTransform(src, dst)
    > Minv = cv2.getPerspectiveTransform(dst,src)
    Masked image area within interest area![Masked image ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/masked_back.png?raw=true)
    > Src point and corresponding dst point![Masked image ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/bird_view.png?raw=true)
    
4. Detect lane pixels and fit to find lane boundary.
	* Firstly, Using histogram and using the maximum value location as start point
	> Histogram and start point![Histogram ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/histogram_start.png?raw=true)
	* Secondly,Using histogram to search lane points row by row
	> Search Point results![Point ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/point_find.jpg?raw=true)
	* Thirdly, need to fit a polynomial to those pixel positions.
	> polynomial Point results![polynomial ](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/plot_scater.png?raw=true)
5. Determine curvature of the lane and vehicle position with respect to center.
	* ym_per_pix = 30/800 # meters per pixel in y dimension
    * xm_per_pix = 3.7/680 # meteres per pixel in x dimension
    > left_fit_cr = np.polyfit(yvals*ym_per_pix, left_fitx*xm_per_pix, 2)
    > right_fit_cr = np.polyfit(yvals*ym_per_pix, right_fitx*xm_per_pix, 2)
	* curvature:
	>   left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
	>   right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])
	>  **curvature 1961.09892301** **
	* center_distance:
	> car_cen = (start_left+start_right)/2.                              
    > center_distance = abs( warped.shape[1]/2. - car_cen)*xm_per_pix
    > **center_distance 0.130588235294m** **
	
6. Warp the detected lane boundaries back onto the original image using pre_calculated inverse matrix
	> newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
	
7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
 > Final results![results](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/final_result.jpg?raw=true)

---
### Video Pipeline Processing:

1. The image processing is successfully implemented to find the lane lines in each frame  the video.Besides, the outputs are generated regarding the radius of curvature of the lane and vehicle position within the lane is also shown in description video.
 * Screen_shot of output video
  > Standard output video
  > Detail Decscription video
 * Output video link
  > Standard output video
  > Detail Decscription video

2. Video processing pipline.
	* The pipline flowchart shown below
	**pipline flowchart**![pipline flowchart](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/video_pipline.png?raw=true)
    Several key points:
    > * In the first few frames of video, the algorithm should perform a search without prior assumptions about where the lines.The start point are calculate based on whole image and row by row search using big window. If missing points, using pre-row point.
    > * Once a high-confidence detection is achieved -- In my case, the first 10 frames and frame whcih line located at allowed threshold as robust, these image would be saved and updated in lanne_class.
    >  * The high confidence information is used in frame line detection:
    >     * For start point, I use start point from preframe to narrow the window size:
    >     **Start point search in first frames**![pipline flowchart](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/imageedit_3_3404131181.jpg?raw=true)
    >     **Start point search in later frames**![Start point search](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/imageedit_6_6127948295.jpg?raw=true)
    >     this not only improve accracy (removing the nosiy info) and increase the search speed (less calculation).
    >     But this also bring larger prob of missing start points, so when missing data or data not believable (distance too large), I use pre_frame start point.
    >      **Start point search in later frames**![Start point search](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/NEW_START%20POINT.png?raw=true)
    >     * For searching of points in each row, I also use smaller window size, reason and advantagement same as above, so is the risk. So I check the points strictly and use pre_info to do inference:
    >       When points missing, I will use the best_fit based on robusrt frames to do prediction;
    >       Then, compare the detected or calculated point with pre_frame and pre_row point, then if they fail, I choose to use point from pre_frame or row.
    >       Detail shown below:
    >       **row point search in later frames**![row point search](https://github.com/shangliy/advanced_lane/blob/master/pipline_images/New_search.png?raw=true)
	
