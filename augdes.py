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

################################################################################
# Code
################################################################################

def main():
    img = cv2.imread(CONCAVE_IM_PATH, 0) # grayscale
    ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((8,8))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('thresh', closed)

if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
