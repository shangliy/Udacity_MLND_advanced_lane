#!usr/bin/python
'''
author: Shanglin Yang (kudoysl@gmail.com)

This script implements whole pipeline on one image 
'''
import glob
import utils
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as lines

import pickle 
from scipy.misc import imshow,imread,imsave





def process_image(image):

    flag_imshow = True
    
    #0. Read in the camera calibration matrix and distortion coefficients
    with open( "camera_cal/wide_dist_pickle.p" ,"rb") as pickfile:
        dist_pickle = pickle.load( pickfile )

    mtx = dist_pickle["mtx"] 
    dist = dist_pickle["dist"]
    
    using_gaussian =True
    #1.undistort image using calibre matrix
    indist = cv2.undistort(image, mtx, dist, None, mtx)
    #1.1 optinal using the gaussian filter to smooth the whole image
    if using_gaussian == True:
        kernel = np.ones((3,3),np.float32)/25
        indist = cv2.GaussianBlur(indist, (3, 3), 0)
        
    #2.Using methods to get binary images

    # Apply each of the thresholding functions
    ksize = 3
    #X-axis Gradient thresh fllter
    gradx = utils.abs_sobel_thresh(indist, orient='x', sobel_kernel=ksize, thresh=(20, 250))

    #Y-axis Gradient thresh fllter
    grady = utils.abs_sobel_thresh(indist, orient='y', sobel_kernel=ksize, thresh=(50, 250))

    #Gradient Magnitude thresh fllter
    mag_binary = utils.mag_thresh(indist, sobel_kernel=ksize, mag_thresh=(40, 250))

    #Gradient direction thresh fllter
    dir_binary = utils.dir_threshold(indist, sobel_kernel=ksize, thresh=(np.pi/6, np.pi/2))
    
    #HLS S-channel thresh fllter
    s_binary = utils.hls_select(indist, thresh=(100, 255))

    # Save images
    diag_gradx = utils.dack_img(gradx)
    diag_grady = utils.dack_img(grady)
    diag_mag = utils.dack_img(mag_binary)
    diag_dir = utils.dack_img(dir_binary)
    diag_s = np.dstack((s_binary,s_binary,s_binary))   
    imsave('gradx.jpg',diag_gradx)
    imsave('grady.jpg',diag_grady)
    imsave('gradient_mag.jpg',diag_mag)
    imsave('gradient_dir.jpg',diag_dir)
    imsave('diag_s.jpg',diag_s)
    
    # Combine all filters to get binary image
    combined = np.zeros_like(dir_binary, dtype=np.uint8)
    combined[ (s_binary == 255)|((((gradx == 1)) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
    img_size = (combined.shape[1], combined.shape[0])
    diag_com = utils.dack_img(combined)
    imsave('diag_com.jpg',diag_com)
    
    #3. Calculate the prospective transform matrix
    # Using masked area,canny edge and hougline to get the corresponding points
    forward_pixel = 480
    src = utils.point_select_ori(combined,forward_pixel)
    offset = 300 # offset for dst points
    offset_y = 0
    dst = np.float32([[offset, offset_y], [img_size[0]-offset, offset_y], 
                                            [img_size[0]-offset, img_size[1]], 
                                            [offset, img_size[1]]])
        
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(combined,M,img_size)

    if (flag_imshow):
        # Plot up the  data
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        
        src_points_x = []
        src_points_y = []
        for poin in src:
            ax1.plot(poin[0], poin[1], 'o', color='red')
            src_points_x.append(poin[0])
            src_points_y.append(poin[1])
        src_points_x.append(src[0][0])
        src_points_y.append(src[0][1])

        dst_points_x = []   
        dst_points_y = [] 
        for poin in dst:
            ax2.plot(poin[0], poin[1], 'o', color='blue')
            dst_points_x.append(poin[0])
            dst_points_y.append(poin[1])
        dst_points_x.append(dst[0][0])
        dst_points_y.append(dst[0][1])
        l = lines.Line2D(src_points_x,src_points_y,linewidth=10)
        r = lines.Line2D(dst_points_x,dst_points_y,linewidth=10,color='red')
        ax1.imshow(combined, cmap='gray')
        ax1.add_line(l)
        ax1.set_title('pre_transform Image', fontsize=50)
        ax2.imshow(warped, cmap='gray')
        ax2.add_line(r)
        ax2.set_title('Bird-view image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    
    #4.Detect lines

    #4.1 Using histogram and using the maximum value location as start point
    distance = warped.shape[0]  
    histogram = np.sum(warped[int(distance/2):,:], axis=0)
        
    max_left = np.max(histogram[50:int(warped.shape[1]/2)])
    max_array = np.where(histogram[50:int(warped.shape[1]/2)] == max_left)
    win_left = np.argmax(histogram[50:int(warped.shape[1]/2)])
    win_left =  50 + np.median(max_array)
    start_left = win_left

    max_right = np.max(histogram[int(warped.shape[1]/2):])
    max_array = np.where(histogram[int(warped.shape[1]/2):] == max_right)
    win_right = int(warped.shape[1]/2)+np.argmax(histogram[int(warped.shape[1]/2):])
    win_right =  int(warped.shape[1]/2) + np.median(max_array)
    start_right = win_right
        
    leftx = np.zeros((distance+1,))
    leftx[distance] = win_left

    rightx = np.zeros((distance+1,))
    rightx[distance] = win_right
    
    y_left_tem = [distance]
    y_right_tem = [distance]
    x_left_tem = [win_left]
    x_right_tem = [win_right]

    plt.plot(range(int(warped.shape[1])),histogram,linewidth=2)
    plt.axvline( win_left,color='g',linewidth=5)
    plt.axvline( win_right,color='r',linewidth=5)
    plt.show()

    right_flag = True
    left_flag = True

     #4.2 Using histogram to search lane points row by row
    win_left_size = 50
    win_right_size = 50
    for yl in range(1,distance+1):
        yvals = distance - yl
        forth_dis = min(100,yvals)
        #Miss points
        if np.sum(np.sum(warped[yvals-forth_dis:yvals,win_left-win_left_size:win_left+win_left_size], axis=0)) < 1:
            
            if  yl < 200: #if eralier miss, the line is not trustable
                left_flag = False  
            if left_flag == True:  #if robust, use degree_1 to do the prediction             
                leftx[yvals] = left_fit_tem[0]*yvals + left_fit_tem[1]
            else: #Not robust, keep the pre point
                leftx[yvals] = win_left
                
        else: # Find points 
            histogram = np.sum(warped[yvals-forth_dis:yvals,win_left- win_left_size:win_left+ win_left_size], axis=0)
            max_left = np.max(histogram)
            max_array = np.where(histogram == max_left)
            lx = win_left- win_left_size + np.median(max_array)
            leftx[yvals] = lx
                
            if  yl < 200: # only remeber the low area images
                x_left_tem.append(win_left)
                y_left_tem.append(yvals)
                left_fit_tem = np.polyfit(y_left_tem, x_left_tem, 1)
            
        win_left = leftx[yvals]

        if np.sum(np.sum(warped[yvals-forth_dis:yvals,win_right-win_right_size:win_right+win_right_size], axis=0)) < 1:
                
            if yl < 200:
                right_flag =False

            if right_flag == True:
                        
                rightx[yvals] = right_fit_tem[0]*yvals + right_fit_tem[1]
            else:
                rightx[yvals] = win_right
                
        else:
                
            histogram = np.sum(warped[yvals-forth_dis:yvals,win_right-win_right_size:win_right+win_right_size], axis=0)
            max_right = np.max(histogram)
            max_array = np.where(histogram == max_right)
            rx = win_right-win_right_size + np.argmax(np.sum(warped[yvals-100:yvals,win_right-win_right_size:win_right+win_right_size], axis=0))
            rx = win_right-win_right_size + np.median(max_array)
            rightx[yvals] = rx
    
            if  yl < 200:
                y_right_tem.append(yvals)
                x_right_tem.append(win_right)
                right_fit_tem = np.polyfit(y_right_tem, x_right_tem, 1)
            
        win_right = rightx[yvals]
        
    #4.3 Ploy points fit lines
    yvals = np.linspace(0, distance, num=distance+1)*1.  # to cover same y-range as image
    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

    if (flag_imshow):
        # Plot up the fake data
        plt.plot(leftx, yvals, 'o', color='red')
        plt.plot(rightx, yvals, 'o', color='blue')
        plt.xlim(0, 1280)
        plt.ylim(0, distance)
        plt.plot(left_fitx, yvals, color='green', linewidth=3)
        plt.plot(right_fitx, yvals, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.show()

    warp_cp = warped.copy()
    warp_new = np.dstack((warp_cp,warp_cp,warp_cp))
    warp_new = warp_new*255
    for i in range(distance+1):
        cx = leftx[i]
        cy = i
        cv2.circle(warp_new,(int(cx),int(cy)),2,(255,0,0),-11)
        cx = rightx[i]
        cy = i
        cv2.circle(warp_new,(int(cx),int(cy)),2,(0,0,255),-11)

    imshow(warp_new)
    

    
    
    #4.4 Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(indist, 1, newwarp, 0.3, 0)
    imshow( result)

    #4.5 Calculate the radius of curvature

    y_eval = np.max(yvals)
    ym_per_pix = 30/800 # meters per pixel in y dimension
    xm_per_pix = 3.7/680 # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(yvals*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals*ym_per_pix, right_fitx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    car_cen = (start_left+start_right)/2.                              
    center_distance = abs( warped.shape[1]/2. - car_cen)*xm_per_pix

    
    
    #print center_distance
    #font = cv2.FONT_HERSHEY_COMPLEX
    #middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    #cv2.putText(middlepanel, 'Estimated lane curvature: ' + str((left_curverad+right_curverad)/2.0), (30, 60), font, 1, (0,255,0), 2)
    #cv2.putText(middlepanel, 'Estimated Meters to the center of the lane: '+ str(center_distance)+"m", (30, 90), font, 1, (0,255,0), 2)

    
    return  result

        
def main():

    # Make a list of test images
    test_images = glob.glob('test_images/*.jpg')
    for idx, fname in enumerate(test_images):
        img = imread(fname)
        process_image(img)
        sys.exit()


if __name__ == '__main__':
    main()
