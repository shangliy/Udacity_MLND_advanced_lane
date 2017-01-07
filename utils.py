import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def calculate_points(img,lines):
    """
    NOTE: This function using lines points try to calculate the right points 
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


    y1_left_fin = 480
    x1_left_fin = x2_left - (y2_left-y1_left_fin)/k_left
    y2_left_fin = img.shape[0]
    x2_left_fin = x2_left - (y2_left-y2_left_fin)/k_left

    

    y1_right_fin = 480
    x1_right_fin = x2_right - (y2_right-y1_right_fin)/k_right
    y2_right_fin = img.shape[0]
    x2_right_fin = x2_right - (y2_right-y2_right_fin)/k_right

    return([x1_left_fin,y1_left_fin],[x2_left_fin,y2_left_fin],[x1_right_fin,y1_right_fin]
           ,[x2_right_fin,y2_right_fin])

def hough_lines_points(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be binary iamge
    Returns four closed points in img 
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    a,b,c,d = calculate_points(img,lines)

    return a,b,c,d

def point_select_ori(binary_img,forward_pixel):

    imshape = binary_img.shape
    vertices = np.array([[(80,imshape[0]-50),(450, forward_pixel), (750, forward_pixel), (imshape[1]-150,imshape[0]-50)]], dtype=np.int32)
    masked_edges = region_of_interest(binary_img,vertices)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(binary_img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(masked_edges, cmap='gray')
    ax2.set_title('Combined image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 60     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    a,b,c,d = hough_lines_points(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    return np.float32([a,c,d,b])


