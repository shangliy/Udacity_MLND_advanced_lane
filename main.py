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

def draw_poly_lines(img, lines, color=[0, 0, 255], thickness=10):
    points_left_x =np.array([])
    points_left_y =np.array([])
    left_num = 0
    points_right_x =np.array([])
    points_right_y =np.array([])
    right_num = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1)<0):
                left_num = left_num + 2;
                points_left_x = np.append(points_left_x, [x1,x2])
                points_left_y = np.append(points_left_y, [y1,y2])
            if ((y2-y1)/(x2-x1)>0):
                right_num = right_num + 2;
                points_right_x = np.append(points_right_x, [x1,x2])
                points_right_y = np.append(points_right_y, [y1,y2])
            print (str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2))

    print (points_left_y.shape)
    print (points_left_x.shape)
    k_left = np.poly1d(np.polyfit(points_left_y, points_left_x, 3))
    k_right = np.poly1d(np.polyfit(points_right_y, points_right_x, 3))

    points_left = np.array([])
    points_right = np.array([])
    num = 0
    for y_tem in range(350,img.shape[0]):
        num = num + 1
        x_left = k_left(y_tem)
        x_right = k_right(y_tem)
        points_left = np.append(points_left, [x_left,y_tem])
        points_right = np.append(points_right, [x_right,y_tem])
    #print (points_left.shape)
    #print (points_right.shape)
    points_left = points_left.reshape(num,2)
    points_right = points_right.reshape(num,2)
    #print (points_left)
    #sys.exit()
    cv2.polylines(img, np.int32([points_left]),False, thickness=thickness, color=color)
    cv2.polylines(img, np.int32([points_right]),False, thickness= thickness, color=color)

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
    y2_left_fin = img.shape[0]
    x2_left_fin = x2_left - (y2_left-y2_left_fin)/k_left

    y1_right_fin = 450
    x1_right_fin = x2_right - (y2_right-y1_right_fin)/k_right
    y2_right_fin = img.shape[0]
    x2_right_fin = x2_right - (y2_right-y2_right_fin)/k_right

    cv2.line(img, (int(x1_left_fin), int(y1_left_fin)), (int(x2_left_fin), int(y2_left_fin)), color, thickness)
    cv2.line(img, (int(x1_right_fin), int(y1_right_fin)), (int(x2_right_fin), int(y2_right_fin)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    line_colorimge = np.dstack((line_img,line_img,line_img))
    draw_lines(line_colorimge, lines)

    return line_colorimge

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
    scale_factor = np.max(gradmag)/255
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
image = mpimg.imread('signs_vehicles_xygrad.png')

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(30, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 100))
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
threshold = 20     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 20 #minimum number of pixels making up a line
max_line_gap = 10    # maximum gap in pixels between connectable line segments

lines_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
#mat_array = cv2.fromarray(masked_edges)
color_edges = np.dstack((masked_edges,masked_edges,masked_edges))
#result = weighted_img(color_edges,lines_image,0.8, 1., 0)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(masked_edges)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(lines_image, cmap='gray')
ax2.set_title('Combined image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
