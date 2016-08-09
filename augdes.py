import cv2
import numpy as np
import os
import math
import time
import copy
import sys

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
SPIRAL_IM_PATH = os.path.join(IM_DIR, 'spiral.jpg')
BIG_BENT_IM_PATH = os.path.join(IM_DIR, 'bigbent.jpg')
SAURAV_IM_PATH = os.path.join(IM_DIR, 'saurav.jpg')
TABLE_IM_PATH = os.path.join(IM_DIR, 'table.jpg')

CHOSEN = TABLE_IM_PATH
LIVE_FLAG = False

GRID_W = 4
GRID_H = 4

MIN_DIST = 5  # NOTE: may have to change this later
MIN_DIST_SQ = MIN_DIST ** 2

BLOCKSIZE = 45
THRESH_WEIGHT = 15

RGB=False

RGB=False

DEBUG=True

################################################################################
# Code
################################################################################

def process(img, design=SPIRAL_IM_PATH):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = grayscale.shape[:2]
    totalarea = width * height

    thresh = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCKSIZE, THRESH_WEIGHT)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contourImage = opened.copy()
    squareContourImage = img.copy()
    avgGridImage = img.copy()
    finalImage = img.copy()
    contours, hierarchy = cv2.findContours(contourImage, cv2.cv.CV_RETR_TREE, cv2.cv.CV_CHAIN_APPROX_NONE)

    cv2.drawContours(contourImage, contours, -1, 255, 3)   

    # holds dicts which contain each cells corner coordinates and centroid
    quadDicts = []
    bigQuad = None
    maxArea = 0

    if not contours:
        return

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
            quad = computeQuad(goodHull, cx, cy)

            if len(quad) == 4:
                quad = sorted(quad, cmp=(lambda a,b : int(np.arctan2(cy - a[1], cx - a[0]) - np.arctan2(cy - b[1], cx - b[0]))))
                quad = np.array([quad]) # expected format by cv2 functions
                area = int(cv2.contourArea(quad))
                #print area, 'vs.', totalarea

                if area <= totalarea * 0.9 and area >= 100:
                    quadDict = {'coords': quad, 'centroid': np.array([cx, cy]), 'area': area}
                    if area > maxArea:
                        maxArea = area
                        bigQuad = quadDict
                    # TODO this lambda causes the polys to cross over themselves
                    # on some cases
                    quadDicts.append(quadDict)
                    #cv2.drawContours(squareContourImage, dpHull, -1, (255,0,0), 2
                    cv2.drawContours(squareContourImage, quad, -1, (0,255,0), 1)
                    #cv2.polylines(squareContourImage, dpHull, True, (0,255,0), 1)
                    cv2.drawContours(squareContourImage, np.matrix([[cx,cy]]),
                                     -1, (0,0,255), 2)



    # adjust for perspective of big quad
    #print 'big quad'
    #print bigQuad
    #print 'dicts1'
    #print quadDicts
    for i in range(len(quadDicts)):
        if quadDicts[i]['coords'][0][0][0] == bigQuad['coords'][0][0][0] and quadDicts[i]['coords'][0][0][1] == bigQuad['coords'][0][0][1]:
            quadDicts.pop(i)
            break
    persp = np.matrix((3,3))
    dst = np.array([[width - 1, height - 1], [0, height - 1], [0,0], [width - 1, 0]], dtype=np.float32)
    persp = cv2.getPerspectiveTransform(bigQuad['coords'][0].astype(np.float32), dst)
    flatgrid = cv2.warpPerspective(img, persp, (width, height))

    if len(quadDicts) < GRID_W*GRID_H:
        return

    def projSortArea(q1, q2):
        return int(q1['area'] - q2['area'])

    def projSortRow(q1, q2):
        cent1 = np.array([q1['centroid']], dtype=np.float32)[None, :, :]
        cent2 = np.array([q2['centroid']], dtype=np.float32)[None, :, :]
        projcent1 = cv2.perspectiveTransform(cent1, persp)
        projcent2 = cv2.perspectiveTransform(cent2, persp)
        return int(projcent1[0][0][0] - projcent2[0][0][0])
        
    def projSortCol(q1, q2):
        cent1 = np.array([q1['centroid']], dtype=np.float32)[None, :, :]
        cent2 = np.array([q2['centroid']], dtype=np.float32)[None, :, :]
        projcent1 = cv2.perspectiveTransform(cent1, persp)
        projcent2 = cv2.perspectiveTransform(cent2, persp)
        return int(projcent1[0][0][1] - projcent2[0][0][1])

    # filter out little quads
    quadDicts = sorted(quadDicts, cmp=projSortArea, reverse=True)
    quadDicts = quadDicts[0:GRID_W*GRID_H]
    
    # sort dicts into row-major order
    
    quadDicts = sorted(quadDicts, cmp=projSortRow)
    for i in range(len(quadDicts)):
        quadDicts[i]['row'] = i/GRID_H
    
    quadDicts = sorted(quadDicts, cmp=projSortCol)
    for i in range(len(quadDicts)):
        quadDicts[i]['col'] = i/GRID_W


    newQuadDicts = [0] * (GRID_W * GRID_H)

    for quadDict in quadDicts:
        newQuadDicts[quadDict['row']*GRID_W + quadDict['col']] = quadDict

    quadDicts = newQuadDicts

    # connect the squares
    for quadDict in quadDicts:
        if DEBUG:
            print quadDict
        if quadDict == 0:
            return
        quadDict['avgcoords'] = np.copy(quadDict['coords'])
  
    for i in range(GRID_H + 1):
        for j in range(GRID_W + 1):
            sum = np.array([0,0])
            num = 0
            if i > 0 and j > 0:
                # bottom right corner of grid to upper left of current
                # intersection
                num += 1
                sum += quadDicts[(i - 1)*GRID_W + j - 1]['coords'][0][0]
            if i > 0 and j < GRID_W:
                # bottom left corner of grid to upper right of current
                # intersection
                num += 1
                sum += quadDicts[(i - 1)*GRID_W + j]['coords'][0][3]
            if i < GRID_H and j < GRID_W:
                # top left corner of grid to lower right of current intersection
                num += 1
                sum += quadDicts[i*GRID_W + j]['coords'][0][2]
            if i < GRID_H and j > 0:
                # top right corner of grid to lower left of current intersection
                num += 1
                sum += quadDicts[i*GRID_W + j - 1]['coords'][0][1]

            # update the coordinates
            avg = sum / num # integer division??? shouldn't be a
                            # problem for this application
            if i > 0 and j > 0:
                # bottom right corner of grid to upper left of current
                # intersection
                quadDicts[(i - 1)*GRID_W + j - 1]['avgcoords'][0][0] = avg
            if i > 0 and j < GRID_W:
                # bottom left corner of grid to upper right of current
                # intersection
                quadDicts[(i - 1)*GRID_W + j]['avgcoords'][0][3] = avg
            if i < GRID_H and j < GRID_W:
                # top left corner of grid to lower right of current intersection
                quadDicts[i*GRID_W + j]['avgcoords'][0][2] = avg
            if i < GRID_H and j > 0:
                # top right corner of grid to lower left of current intersection
                quadDicts[i*GRID_W + j - 1]['avgcoords'][0][1] = avg

    for k in range(len(quadDicts)):
        quadDict = quadDicts[k]
        cv2.drawContours(avgGridImage, quadDict['avgcoords'], -1, (0,255,0), 1)
        cv2.drawContours(avgGridImage, np.matrix([quadDict['centroid']]), -1, (0,0,255/(len(quadDicts))*(k+1)), 5)
    cv2.drawContours(avgGridImage, np.matrix([[187, 131], [0,0], [0,100]]), -1, (0,255,255), 7)
        
    # array of corners in row major order of size (GRID_W+1) *(GRID_H+1), where the first two indices are x,y and the third is count 
    #corners = [[0,0,0]] * ((GRID_W+1) * (GRID_H + 1))
    #for i in range(len(quadDicts)):
    #    dict = quadDicts[i]
    #    for j in range(len(dict['coords'])):
    #        pass

    # calc average corner position
    #for i in range(len(corners)):
    #    corners[i] = corners[i][0] / corners[i][2]

    spiral = cv2.imread(design, cv2.CV_LOAD_IMAGE_COLOR)
    sheight, swidth = spiral.shape[:2]
    dst = np.zeros((width, height, 3))

    first = True
    for r in range(GRID_H):
        for c in range(GRID_W):
            base = np.array([[swidth/GRID_W,sheight/GRID_H], [0,sheight/GRID_H], [0,0], [swidth/GRID_W, 0]], dtype=np.float32)
            br = [(c+1)*swidth/GRID_W, (r+1)*sheight/GRID_H]
            ur = [c*swidth/GRID_W, (r+1)*sheight/GRID_H]
            ul = [c*swidth/GRID_W, r*sheight/GRID_H]
            bl = [(c+1)*swidth/GRID_W, r*sheight/GRID_H]
            design_coords = np.array([br, ur, ul, bl], dtype=np.float32)
            #print '(',r,',',c,')'
            #print design_coords
            #print quadDicts[r*GRID_W + c]['avgcoords'][0].astype(np.float32)
            persp = np.matrix((3,3))
            persp = cv2.getPerspectiveTransform(base, quadDicts[r*GRID_W + c]['avgcoords'][0].astype(np.float32))
            monkey = cv2.warpPerspective(spiral[ul[0]:br[0],ul[1]:ur[1]], persp, (width, height))
            #print monkey.shape[:2]
            #print dst.shape[:2]
            if first:
                first = False
                dst = monkey
            else:
                dst = cv2.add(dst,monkey)

    ret, masky = cv2.threshold(dst, 2, 255, cv2.THRESH_BINARY_INV)
    cleared = cv2.bitwise_and(img, img, mask=(masky[:,:,0] + masky[:,:,1] + masky[:,:,2])/3)
    added = cleared + dst
    
    # display results
    if not LIVE_FLAG:
        cv2.imshow('original', img)
        if DEBUG:
            cv2.imshow('thresh', thresh)
            cv2.imshow('opened', opened)
            cv2.imshow('contours', contourImage)
            cv2.imshow('squares', squareContourImage)
            cv2.imshow('whoa', flatgrid)
            cv2.imshow('avg', avgGridImage)
        cv2.imshow('spiral', spiral)
        cv2.imshow('final', added)
        ypos = 100
        xpos = 10
        delta = 300
        cv2.moveWindow('original', xpos, ypos)
        if DEBUG:
            cv2.moveWindow('thresh', xpos + delta, ypos)
            cv2.moveWindow('opened', xpos + delta*2, ypos)
            cv2.moveWindow('contours', xpos + delta*3, ypos)
            cv2.moveWindow('squares', xpos + delta*4, ypos)
            cv2.moveWindow('whoa', xpos + delta*5, ypos)
            cv2.moveWindow('avg', xpos, ypos)
        cv2.moveWindow('spiral', xpos + delta, ypos)
        cv2.moveWindow('final', xpos + delta*2, ypos)
    else:
        #cv2.imshow('thresh', opened)
        cv2.imshow('processed', added)
        ypos = 10
        xpos = 10
        delta = 300
        cv2.moveWindow('processed', xpos, ypos)
        #cv2.moveWindow('thresh', xpos, ypos+delta*2)
    return added
        

# there should be at least 4 points in the hull before passing it to this function
def computeQuad(hull, cx, cy):
    # farthest to closest
    sortedHull = sorted(hull, cmp=(lambda x,y : distSq(x,[cx,cy]) - distSq(y,[cx,cy])),reverse = True)
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
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy


if __name__ == '__main__':
    if LIVE_FLAG:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('processed', cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read() #read a frame
            if len(sys.argv) == 1:
                process(frame)
            else:
                process(frame, sys.argv[1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        img = None
        img = cv2.imread(CHOSEN, cv2.CV_LOAD_IMAGE_COLOR)
        if len(sys.argv) == 1:
            process(img)
        else:
            process(img, sys.argv[1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

