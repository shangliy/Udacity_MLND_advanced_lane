import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def draw_lines(img, lines, color=[0, 0, 255], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    k_left = 0
    k_right = 0
    line_left = 0
    line_right = 0
    x1_left = 0
    y1_left = 0
    x2_left = 0
    y2_left = 0

    x1_right = 0
    y1_right = 0
    x2_right = 0
    y2_right = 0

    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1)<0):
                x1_left = x1_left + x1
                y1_left = y1_left + y1
                x2_left = x2_left + x2
                y2_left = y2_left + y2

                line_left = line_left + 1
            elif ((y2-y1)/(x2-x1)>0):
                x1_right = x1_right + x1
                y1_right = y1_right + y1
                x2_right = x2_right + x2
                y2_right = y2_right + y2
                line_right = line_right + 1

    x1_left = x1_left/line_left
    y1_left = y1_left/line_left
    x2_left = x2_left/line_left
    y2_left = y2_left/line_left

    x1_right = x1_right/line_right
    y1_right = y1_right/line_right
    x2_right = x2_right/line_right
    y2_right = y2_right/line_right

    k_left = (y2_left-y1_left)/(x2_left-x1_left)
    k_right = (y2_right-y1_right)/(x2_right-x1_right)


    y1_left_fin = 450
    x1_left_fin = x2_left - (y2_left-y1_left_fin)/k_left
    y2_left_fin = img.shape[0]-100
    x2_left_fin = x2_left - (y2_left-y2_left_fin)/k_left

    y1_right_fin = 450
    x1_right_fin = x2_right - (y2_right-y1_right_fin)/k_right
    y2_right_fin = img.shape[0]-100
    x2_right_fin = x2_right - (y2_right-y2_right_fin)/k_right

    return([x1_left_fin,y1_left_fin],[x2_left_fin,y2_left_fin],[x1_right_fin,y1_right_fin]
           ,[x2_right_fin,y2_right_fin])

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    line_colorimge = np.dstack((line_img,line_img,line_img))
    a,b,c,d = draw_lines(line_colorimge, lines)

    return a,b,c,d

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255.
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    # Here I'm suppressing annoying error messages
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

    s = hls[:,:,2]
    s = 255.0 * s/np.max(s)

    retval, binary_output = cv2.threshold(s.astype('uint8'),thresh[0],thresh[1],cv2.THRESH_BINARY)
    return binary_output



# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements
# Read in an image
image = mpimg.imread('test3.jpg')

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(40, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(np.pi/6, np.pi/4))
s_binary = hls_select(image, thresh=(90, 255))

combined = np.zeros_like(dir_binary, dtype=np.uint8)
combined[ (s_binary == 255)|((((gradx == 1)) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1


imshape = combined.shape
vertices = np.array([[(0,imshape[0]-100),(600, 450), (800, 450), (imshape[1]-400,imshape[0]-100)]], dtype=np.int32)
masked_edges = region_of_interest(combined,vertices)
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 30     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments

a,b,c,d = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
#mat_array = cv2.fromarray(masked_edges)
src =  np.float32([a,c,d,b])
offset = 100 # offset for dst points
dst = np.float32([[offset, 0], [imshape[0]-offset, 0],
                                     [imshape[0]-offset, imshape[1]],
                                     [offset, imshape[1]]])

print(src)
print(dst)
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst,src)
warped = cv2.warpPerspective(combined,M,imshape)
#color_edges = np.dstack((masked_edges,masked_edges,masked_edges))
#result = weighted_img(color_edges,lines_image,0.8, 1., 0)


histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)


win_left = np.argmax(histogram[:int(warped.shape[1]/2)])
win_right = int(warped.shape[1]/2)+np.argmax(histogram[int(warped.shape[1]/2):])
print (win_left)
print (win_right)

leftx = np.zeros((1001,))
leftx[0] = win_left

rightx = np.zeros((1001,))
rightx[0] = win_right

for yl in range(1,1000):
    yvals = warped.shape[0] - yl



    if np.max(np.sum(warped[yvals-10:yvals,win_left-15:win_left+15], axis=0)) < 1:

        leftx[yl] = win_left
        win_left = win_left
    else:
        lx = win_left-15 + np.argmax(np.sum(warped[yvals-10:yvals,win_left-15:win_left+15], axis=0))
        leftx[yl] = lx
        win_left = lx


    if np.max(np.sum(warped[yvals-10:yvals,win_right-15:win_right+15], axis=0)) < 1:
        rightx[yl] = win_right
        win_right = win_right
    else:
        rx = win_right-15 + np.argmax(np.sum(warped[yvals-10:yvals,win_right-15:win_right+15], axis=0))
        rightx[yl] = rx
        win_right = rx



leftx = leftx[::-1]
rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

yvals = np.linspace(0, 1000, num=1001)*1.  # to cover same y-range as image
left_fit = np.polyfit(yvals, leftx, 2)
left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
right_fit = np.polyfit(yvals, rightx, 2)
right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

# Plot up the fake data
plt.plot(leftx, yvals, 'o', color='red')
plt.plot(rightx, yvals, 'o', color='blue')
plt.xlim(0, 800)
plt.ylim(0, 1100)
plt.plot(left_fitx, yvals, color='green', linewidth=3)
plt.plot(right_fitx, yvals, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
plt.show()

y_eval = np.max(yvals)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                             /np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                /np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)

# Create an image to draw the lines on
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
result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
plt.imshow(result)


ym_per_pix = 30/1000 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meteres per pixel in x dimension

left_fit_cr = np.polyfit(yvals*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(yvals*ym_per_pix, rightx*xm_per_pix, 2)
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                             /np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                /np.absolute(2*right_fit_cr[0])
# Now our radius of curvature is in meters
print(left_curverad, 'm', right_curverad, 'm')
# Example values: 3380.7 m    3189.3 m



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(masked_edges)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(warped, cmap='gray')
ax2.set_title('Combined image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
