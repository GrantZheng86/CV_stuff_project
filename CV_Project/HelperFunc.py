import cv2
import numpy as np
import os
import glob
from pip._vendor.distlib._backport.tarfile import TOREAD


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