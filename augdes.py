import cv2
import numpy as np
import os

################################################################################
# Constants
################################################################################

IM_DIR = 'images'
FLAT_IM_PATH = os.path.join(IM_DIR, 'flat.jpg')
CONCAVE_IM_PATH = os.path.join(IM_DIR, 'concave.jpg')
CONVEX_IM_PATH = os.path.join(IM_DIR, 'convex.jpg')
FLAT2_IM_PATH = os.path.join(IM_DIR, 'flat2.jpg')
CONCAVE2_IM_PATH = os.path.join(IM_DIR, 'concave2.jpg')
CONVEX2_IM_PATH = os.path.join(IM_DIR, 'convex2.jpg')

################################################################################
# Code
################################################################################

def main():
    img = cv2.imread(CONCAVE2_IM_PATH, cv2.CV_LOAD_IMAGE_COLOR) # grayscale
    grayscale = cv2.imread(CONCAVE2_IM_PATH, 0) # grayscale
    
    ret, thresh = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    #kernel = np.ones((8,8))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contourImage = closed.copy()
    squareContourImage = img.copy()
    contours, hierarchy = cv2.findContours(contourImage, cv2.cv.CV_RETR_TREE, cv2.cv.CV_CHAIN_APPROX_NONE)

    cv2.drawContours(contourImage, contours, -1, 255, 3)   
 
    squares = []

    if not contours:
        print "No contours!"
    else:
        # test each contour
        print len(contours)
        for contour in contours:
            # approximate contour with accuracy proportional
            # to the contour perimeter

            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            resultDP = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*0.02, True)
            dpHull = cv2.convexHull(resultDP)
            hull = cv2.convexHull(contour)

            # square contours should have 4 vertices after approximation
            # relatively large area (to filter out noisy contours)
            # and be convex.
            # Note: absolute value of an area is used because
            # area may be positive or negative - in accordance with the
            # contour orientation
   
    	    cv2.drawContours(squareContourImage, dpHull, -1, (255,0,0), 2)
    	    cv2.drawContours(squareContourImage, hull, -1, (0,255,0), 2)
    	    cv2.drawContours(squareContourImage, np.matrix([[cx,cy]]), -1, (0,0,255), 2)

            

    #print squares
    
    for square in squares:
        # print square
        cv2.rectangle(squareContourImage, tuple(square[0][0]), tuple(square[2][0]), (255, 0, 0))

    cv2.imshow('original', img)
    cv2.imshow('thresh', thresh)
    cv2.imshow('closed', closed)
    cv2.imshow('contours', contourImage)
    cv2.imshow('squares', squareContourImage)
    ypos = 100
    xpos = 10
    delta = 300
    cv2.moveWindow('original', xpos, ypos)
    cv2.moveWindow('thresh', xpos + delta, ypos)
    cv2.moveWindow('closed', xpos + delta*2, ypos)
    cv2.moveWindow('contours', xpos + delta*3, ypos)
    cv2.moveWindow('squares', xpos + delta*4, ypos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
