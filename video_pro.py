import os
import sys
import argparse

import numpy as np
import cv2
import math 

import pickle

import utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.misc import imshow,imsave
from moviepy.editor import VideoFileClip

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # Start point
        self.start = None



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', dest='video_input', type=str,\
                        help='The path of input video')
parser.add_argument('-o', dest='video_output', type=str,\
                        help='The path of output video')
args = parser.parse_args()
print(args)

video_input = args.video_input
video_output = args.video_output
frame = 0 
frame_cut = 10
forward_pixel = 480
flag_imshow = False

line_left = Line()
line_right = Line()

with open( "camera_cal/wide_dist_pickle.p" ,"rb") as pickfile:
    dist_pickle = pickle.load( pickfile )

mtx = dist_pickle["mtx"] 
dist = dist_pickle["dist"]


def process_image(image):
    global mtx,dist
    global frame
    global M,Minv
    global flag_imshow
    global line_left,line_right

    if frame>-1:
        flag_imshow = True
        #imsave('test_'+str(frame)+'.jpg',image)
    #flag_imshow = False
    using_gaussian =True
    #1.undistort image using calibre matrix
    indist = cv2.undistort(image, mtx, dist, None, mtx)
    #1.1 optinal 
    if using_gaussian == True:

        kernel = np.ones((3,3),np.float32)/25
        indist = cv2.GaussianBlur(indist, (3, 3), 0)
        
    #2.Using method to get binary images
    # Apply each of the thresholding functions
    ksize = 3
    gradx = utils.abs_sobel_thresh(indist, orient='x', sobel_kernel=ksize, thresh=(20, 250))
    grady = utils.abs_sobel_thresh(indist, orient='y', sobel_kernel=ksize, thresh=(50, 250))
    mag_binary = utils.mag_thresh(indist, sobel_kernel=ksize, mag_thresh=(40, 250))
    dir_binary = utils.dir_threshold(indist, sobel_kernel=ksize, thresh=(np.pi/6, np.pi/4))
    s_binary = utils.hls_select(indist, thresh=(120, 255))

    combined = np.zeros_like(dir_binary, dtype=np.uint8)
    combined[ (s_binary == 255)|((((gradx == 1)) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
    img_size = (combined.shape[1], combined.shape[0])

    if (flag_imshow):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Combined image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    
    #3.For first images, need to calculate the prospective transform matrix
    if (frame == 0):
        src = utils.point_select_ori(combined,forward_pixel)
        offset = 300 # offset for dst points
        offset_y = 250
        dst = np.float32([[offset, offset_y], [img_size[0]-offset, offset_y], 
                                            [img_size[0]-offset, img_size[1]], 
                                            [offset, img_size[1]]])
        #print(src)
        #print(dst)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst,src)
        warped = cv2.warpPerspective(combined,M,img_size)

        if (flag_imshow):
        # Plot up the  data

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
                
            for poin in src:
                ax1.plot(poin[0], poin[1], 'o', color='red')

            for poin in dst:
                ax2.plot(poin[0], poin[1], 'o', color='blue')
            ax1.imshow(combined)
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(warped, cmap='gray')
            ax2.set_title('Combined image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()

    warped = cv2.warpPerspective(combined,M,img_size)
    #color_edges = np.dstack((masked_edges,masked_edges,masked_edges))
    #result = weighted_img(color_edges,lines_image,0.8, 1., 0)

    if (flag_imshow):
        # Plot up the  data
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(combined)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(warped, cmap='gray')
        ax2.set_title('Combined image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    
    #4.Detect lines

    if (frame < frame_cut):
        distance = warped.shape[0]
        histogram = np.sum(warped[distance/2:,:], axis=0)

        
        win_left = np.argmax(histogram[:int(warped.shape[1]/2)])
        win_right = int(warped.shape[1]/2)+np.argmax(histogram[int(warped.shape[1]/2):])
        start_left = win_left
        start_right = win_right
        #print(start_left)
        #print(line_left.start)
        #print (win_left)
        #print (win_right)

        leftx = np.zeros((distance+1,))
        leftx[distance] = win_left

        rightx = np.zeros((distance+1,))
        rightx[distance] = win_right

        for yl in range(1,distance+1):
            yvals = distance - yl

            if np.sum(np.sum(warped[yvals-40:yvals,win_left-15:win_left+15], axis=0)) < 1:

                leftx[yvals] = win_left
                win_left = win_left
            else:
                histogram = np.sum(warped[yvals-40:yvals,win_left-15:win_left+15], axis=0)
                max_left = np.max(histogram)
                max_array = np.where(histogram == max_left)
                lx = win_left-15 + np.argmax(np.sum(warped[yvals-40:yvals,win_left-15:win_left+15], axis=0))
                lx = win_left-15 + np.median(max_array)
                leftx[yvals] = lx
                win_left = lx


            if np.sum(np.sum(warped[yvals-40:yvals,win_right-15:win_right+15], axis=0)) < 1:
                rightx[yvals] = win_right
                win_right = win_right
            else:
                histogram = np.sum(warped[yvals-40:yvals,win_right-15:win_right+15], axis=0)
                max_right = np.max(histogram)
                max_array = np.where(histogram == max_right)
                rx = win_right-15 + np.argmax(np.sum(warped[yvals-40:yvals,win_right-15:win_right+15], axis=0))
                rx = win_right-15 + np.median(max_array)
                rightx[yvals] = rx
                win_right = rx
        

    else:

        distance = warped.shape[0]
        histogram = np.sum(warped[distance/2:,:], axis=0)
        
        left_start_av = line_left.start
        right_start_av = line_right.start
        #print ('left_start_av',left_start_av)
        #win_left = left_start_av-15 + np.argmax(histogram[left_start_av-15:left_start_av+15])
        #win_right =right_start_av -15 + int(warped.shape[1]/2)+np.argmax(histogram[right_start_av-15:right_start_av+15])

        if np.sum(histogram[left_start_av-15:left_start_av+15]) < 1:
            win_left = left_start_av
            #line_left.detected = False
            
        else:
            win_left = left_start_av-15 + np.argmax(histogram[left_start_av-15:left_start_av+15])
            if abs(win_left-left_start_av) > 15:
                win_left = left_start_av
                #line_left.detected = False
            else:
                left_start = win_left
                
        if np.sum(histogram[right_start_av-15:right_start_av+15]) < 1:
            win_right = right_start_av
            #line_right.detected = False
            
        else:
            win_right =right_start_av -15 +np.argmax(histogram[right_start_av-15:right_start_av+15])
            if abs(win_right-right_start_av) > 15:
                win_right = right_start_av
                #line_right.detected = False
            else:
                right_start = win_right
                

        leftx = np.zeros((distance+1,))
        leftx[distance] = win_left
        #print (leftx)

        rightx = np.zeros((distance+1,))
        rightx[distance] = win_right

        start_left = win_left
        start_right = win_right
        #print ('start_left',start_left)

        #print (win_left)
        
        #print (win_right)
        window_size_left = 20
        window_size_right = 20
        for yl in range(1,distance+1):
            yvals = distance - yl
            
            if np.sum(np.sum(warped[yvals-20:yvals,win_left-window_size_left:win_left+window_size_left], axis=0)) < 1:

                leftx[yvals] = line_left.best_fit[0]*yvals**2 + line_left.best_fit[1]*yvals + line_left.best_fit[2]
                win_left = leftx[yvals]
                #window_size_left = 100
                #line_left.detected = False
            else:
                histogram = np.sum(warped[yvals-40:yvals,win_left-window_size_left:win_left+window_size_left], axis=0)
                max_left = np.max(histogram)
                max_array = np.where(histogram == max_left)
                lx = win_left-window_size_left + np.argmax(np.sum(warped[yvals-20:yvals,win_left-window_size_left:win_left+window_size_left], axis=0))
                lx = win_left-window_size_left + np.median(max_array)
                leftx[yvals] = lx
                win_left = lx
            if (abs(line_left.recent_xfitted[yvals]-leftx[yvals]))>50:
                leftx[yvals] = line_left.recent_xfitted[yvals]

            #print ([yvals],leftx[yvals])

            if np.sum(np.sum(warped[yvals-20:yvals,win_right-window_size_right:win_right+window_size_right], axis=0)) < 1:
                
                rightx[yvals] = line_right.best_fit[0]*yvals**2 + line_right.best_fit[1]*yvals + line_right.best_fit[2]
                win_right = rightx[yvals]
                #window_size_right = 100
                #line_right.detected = False
            else:
                histogram = np.sum(warped[yvals-20:yvals,win_right-window_size_right:win_right+window_size_right], axis=0)
                max_right = np.max(histogram)
                max_array = np.where(histogram == max_right)
                rx = win_right-window_size_right + np.argmax(np.sum(warped[yvals-20:yvals,win_right-window_size_right:win_right+window_size_right], axis=0))
                rx = win_right-window_size_right + np.median(max_array)
                rightx[yvals] = rx
                win_right = rx
            if (abs(line_right.recent_xfitted[yvals]-rightx[yvals]))>50:
                rightx[yvals] = line_right.recent_xfitted[yvals]
        
            
        


    #leftx = leftx[::-1]
    #rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        
    yvals = np.linspace(0, distance, num=distance+1)*1.  # to cover same y-range as image
    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

    #print('line_left.start',line_left.start)

    if ( line_left.detected  ):
        line_left.recent_xfitted = left_fitx
        line_left.bestx *= frame
        line_left.bestx += left_fitx*frame
        line_left.best_fit *= frame
        line_left.best_fit += left_fit*frame
        line_left.start *= frame
        line_left.start += start_left*frame
        line_left.bestx /= (frame+frame)
        line_left.best_fit /= (frame+frame)
        line_left.start /= (frame+frame)
    else:
        line_left.bestx = left_fitx
        line_left.best_fit = left_fit
        line_left.start = start_left
        line_left.detected = True

    
    if ( line_right.detected  ):
        line_right.recent_xfitted = right_fitx
        line_right.bestx *= frame
        line_right.bestx += right_fitx*frame
        line_right.best_fit *= frame
        line_right.best_fit += frame*right_fit
        line_right.start *= frame
        line_right.start += start_right*frame
        line_right.bestx /= (frame+frame)
        line_right.best_fit /= (frame+frame)
        line_right.start /= (frame+frame)
    else:
        line_right.bestx = right_fitx
        line_right.best_fit = right_fit
        line_right.start = start_right
        line_right.detected = True

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
        cv2.circle(warp_new,(int(cx),int(cy)),10,(255,0,0),-11)
        cx = rightx[i]
        cy = i
        cv2.circle(warp_new,(int(cx),int(cy)),10,(0,0,255),-11)

    #imshow(warp_new)
    

    y_eval = np.max(yvals)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                                        /np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                            /np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    left_fitx = 0.9*left_fitx + 0.1*line_left.bestx
    right_fitx = 0.9*right_fitx + 0.1*line_right.bestx
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(indist, 1, newwarp, 0.3, 0)

    ym_per_pix = 30/800 # meters per pixel in y dimension
    xm_per_pix = 3.7/680 # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(yvals*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')


    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated left lane curvature:' + str(left_curverad), (30, 40), font, 1, (255,0,0), 2)
    cv2.putText(middlepanel, 'Estimated left lane curvature:' + str(left_curverad), (30, 40), font, 1, (255,0,0), 2)
    cv2.putText(middlepanel, 'Estimated Meters right of center: ERROR!', (30, 90), font, 1, (255,0,0), 2)

    def dack_img(img):
        img_out = np.dstack((img,img,img))
        img_out = img_out * 255
        return img_out
    diag_gradx = dack_img(gradx)
    diag_grady = dack_img(grady)
    diag_mag = dack_img(mag_binary)
    diag_dir = dack_img(dir_binary)
    diag_s = np.dstack((s_binary,s_binary,s_binary))
    diag_com = dack_img(combined)
    # assemble the screen example
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = result
    diagScreen[0:240, 1280:1600] = cv2.resize( diag_gradx, (320,240), interpolation=cv2.INTER_AREA) 
    diagScreen[0:240, 1600:1920] = cv2.resize( diag_grady, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1280:1600] = cv2.resize(diag_mag, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1600:1920] = cv2.resize(diag_dir, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[480:720, 1280:1600] = cv2.resize(diag_s, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[480:720, 1600:1920] = cv2.resize(diag_com, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[720:840, 0:1280] = middlepanel
    diagScreen[720:1080, 1280:1920] = cv2.resize(warp_new, (640,360), interpolation=cv2.INTER_AREA)


        


    if (flag_imshow):
        imshow(result)
    frame = frame + 1
    #imsave('diag_'+str(frame)+'.jpg',diagScreen)
    
    return diagScreen

        


clip1 = VideoFileClip(video_input)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(video_output, audio=False)
