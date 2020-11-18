import cv2
import numpy as np
import os
import glob
from pip._vendor.distlib._backport.tarfile import TOREAD
from pip._vendor.chardet import detect


CALIBRATION_PATH = "Images\\Camera Calibration\\"
STEREO_PATH = "Images\\Stereo Images\\"


def calibrateCamera():
    """
    This camera calibration code was adapted from OpenCV's document page
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    """
    boardH = 7
    boardW = 7
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # objPts are 3D points in real world on the chess board
    # TODO: Measure the actual dimension of the square and put it here
    objPts = np.zeros((boardH*boardW, 3), np.float32)
    objPts[:, :2] = np.mgrid[0:boardH, 0:boardW].T.reshape(-1, 2)
    
    objectPoints = []
    imagePoints = []
    
    calibrationImages = glob.glob(os.path.join(CALIBRATION_PATH, "*.JPG"))
    assert(len(calibrationImages) > 0)
    
    
    #for eachImage in calibrationImages:
    for i in range(len(calibrationImages)):
        
        currImg = cv2.imread(calibrationImages[i])
        gray = cv2.cvtColor(currImg, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (boardH, boardW), None)
        
        if ret:
            objectPoints.append(objPts)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imagePoints.append(corners2)
            
            
            #img = cv2.drawChessboardCorners(currImg, (boardH, boardW), corners2, ret)
            #cv2.imshow('Chess board Calibration', cv2.resize(img, (1280,720)))
            #cv2.waitKey(0)
            
            
    cv2.destroyAllWindows()
    
    # gray.shape[::-1] makes the dimension in reverse order, this method uses different
    # dimension convention than the image one
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints,
                                                       gray.shape[::-1], None, None)
    
    saveCameraCalibration(mtx, dist)

def saveCameraCalibration(mtx, dist):
    mtx = np.array(mtx)
    dist = np.array(dist)
    np.save('M.npy', mtx)
    np.save('D.npy', dist)
    
def loadCameraCalibration():
    mtx = np.load("M.npy")
    dist = np.load("D.npy")
    

    return mtx, dist

def findMatchingPoints(im1, im2, show_image = False):
    detector = cv2.ORB_create(nfeatures=2000, nlevels=8, firstLevel=0, patchSize=31, edgeThreshold=31)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    im1_key_pts, im1_desc = detector.detectAndCompute(im1_gray, mask = None)
    im2_key_pts, im2_desc = detector.detectAndCompute(im2_gray, mask = None)
    
    matches = matcher.knnMatch(im1_desc, im2_desc, k = 2)
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    matches = good
    im1Pts = []
    im2Pts = []
    for match in matches:
        im1Pts.append(im1_key_pts[match.queryIdx].pt)
        im2Pts.append(im2_key_pts[match.trainIdx].pt)
        
    bgr_matches = cv2.drawMatches(img1=im1, keypoints1=im1_key_pts,
                                  img2=im2, keypoints2=im2_key_pts,
                                  matches1to2=matches, matchesMask=None, outImg=None)
    
    if show_image:
        cv2.namedWindow("All matches", cv2.WINDOW_NORMAL) 
        cv2.imshow("All matches", bgr_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return im1Pts, im2Pts