import cv2
import numpy as np
import os
import math

################################################################################
# Constants
################################################################################

IM_DIR = 'images'
FLAT_IM_PATH = os.path.join(IM_DIR, 'flat.jpg')
CONCAVE_IM_PATH = os.path.join(IM_DIR, 'concave.jpg')
CONVEX_IM_PATH = os.path.join(IM_DIR, 'convex.jpg')
FLAT2_IM_PATH = os.path.join(IM_DIR, 'flat2.jpg')
CONCAVE2CROP_IM_PATH = os.path.join(IM_DIR, 'concave2crop.jpg')
CONVEX2_IM_PATH = os.path.join(IM_DIR, 'convex2.jpg')

CHOSEN = CONCAVE2CROP_IM_PATH

LIVE_FLAG = False

## CONSTANT
MIN_DIST = 10 #40  # NOTE: may have to change this later
MIN_DIST_SQ = MIN_DIST ** 2

################################################################################
# Code
################################################################################

def process(img):
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = grayscale.shape[:2]
    totalarea = width * height
    print totalarea
    #ret, thresh = cv2.threshold(grayscale, 160, 255, cv2.THRESH_BINARY)
    #ret, thresh = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)

    ## CONSTANT
    blocksize = 45
    threshWeight = 5

    thresh = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, threshWeight)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contourImage = opened.copy()
    squareContourImage = img.copy()
    contours, hierarchy = cv2.findContours(contourImage, cv2.cv.CV_RETR_TREE, cv2.cv.CV_CHAIN_APPROX_NONE)

    cv2.drawContours(contourImage, contours, -1, 255, 3)   
 
    squares = []

    if not contours:
        return
    else:
        # test each contour
        for contour in contours:
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
 
            ## CONSTANT
            polygonConstant = 0.02
            resultDP = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*polygonConstant, True)
            dpHull = cv2.convexHull(resultDP)


            if len(dpHull) >= 4:
                goodHull = map(lambda x : x[0], dpHull)
                # compute the quadrilaterals
                quads = computeQuads(goodHull, cx, cy)

                if len(quads) == 4:
                    quads = sorted(quads, cmp=(lambda a,b : int(np.arctan2(cy - a[1], cx - a[0]) - np.arctan2(cy - b[1], cx - b[0]))))
                    if cv2.contourArea(np.array([quads])) <= totalarea/2: #totalarea/4:
                        # TODO this lambda causes the polys to cross over themselves on some cases
                        badQuads = np.array([quads])
                        #cv2.drawContours(squareContourImage, dpHull, -1, (255,0,0), 2)
                        cv2.drawContours(squareContourImage, badQuads, -1, (0,255,0), 1)
                        #cv2.polylines(squareContourImage, dpHull, True, (0,255,0), 1)
                        cv2.drawContours(squareContourImage, np.matrix([[cx,cy]]), -1, (0,0,255), 2)

    if not LIVE_FLAG:
        cv2.imshow('original', img)
        cv2.imshow('thresh', thresh)
        cv2.imshow('opened', opened)
        cv2.imshow('contours', contourImage)
        cv2.imshow('squares', squareContourImage)
        ypos = 100
        xpos = 10
        delta = 300
        cv2.moveWindow('original', xpos, ypos)
        cv2.moveWindow('thresh', xpos + delta, ypos)
        cv2.moveWindow('opened', xpos + delta*2, ypos)
        cv2.moveWindow('contours', xpos + delta*3, ypos)
        cv2.moveWindow('squares', xpos + delta*4, ypos)
    else:
        #cv2.imshow('thresh', opened)
        cv2.imshow('processed', squareContourImage)
        ypos = 10
        xpos = 10
        delta = 300
        cv2.moveWindow('processed', xpos, ypos)
        #cv2.moveWindow('thresh', xpos, ypos+delta*2)
        


# there should be at least 4 points in the hull
def computeQuads(hull, cx, cy):
    # farthest to closest
    sortedHull = sorted(hull, cmp=(lambda x,y : distSq(x,[cx,cy]) - distSq(y,[cx,cy])), reverse = True)
    candidates = []
    i = 0
    while len(candidates) < 4 and i < len(sortedHull):
        point = sortedHull[i]
        farEnough = True
        for cand in candidates:
            if distSq(point, cand) < MIN_DIST_SQ:
                farEnough = False
                break
        if farEnough:
            candidates.append(point)
        i += 1

    if len(candidates) < 4:
        return []
    return candidates


def distSq(p1, p2):
    d1 = p1[0] - p2[0]
    d2 = p1[1] - p2[1]
    return d1*d1 + d2*d2


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))



if __name__ == '__main__':
    if LIVE_FLAG:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('processed', cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read() #read a frame
            process(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        img = cv2.imread(CHOSEN, cv2.CV_LOAD_IMAGE_COLOR)
        process(img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

