## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project is to use advanced technologies to solve lane detection problem.

### Description:
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
  
3. Apply a perspective transform to rectify binary image ("birds-eye view"). 
4. Detect lane pixels and fit to find lane boundary.
5. Determine curvature of the lane and vehicle position with respect to center.
6. Warp the detected lane boundaries back onto the original image.
7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

The images in `test_images` are for testing your pipeline on single frames.  The video called `project_video.mp4` is the video your pipeline should work well on.  `challenge_video.mp4` is an extra (and optional) challenge for you if you want to test your pipeline.

If you're feeling ambitious (totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!# advanced_lane
