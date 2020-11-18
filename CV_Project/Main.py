import HelperFunc
import cv2
import os.path
import numpy as np

if __name__ == '__main__':
    # NOTE: Only run camera calibration when need to re-calibrate camera
    #       This function will take a long time to run for very large image
    
    # NOTE: Do not try to undistort the camera, peraxial images can be solved very well, 
    # but does not work good for other cases.
    cam_matrix, distortion = HelperFunc.loadCameraCalibration()
    currDir = os.getcwd()
    im1 = cv2.imread(os.path.join(currDir, 
                                  "Images\\Stereo Images\\0in.JPG"))
    im2 = cv2.imread("Images\\Stereo Images\\6in.JPG")

    im1_pts, im2_pts = HelperFunc.findMatchingPoints(im1, im2)
    im1_pts = np.float32(im1_pts)
    im2_pts = np.float32(im2_pts)
    
    # Finding the fundamental matrix
    F, mask = cv2.findFundamentalMat(im1_pts, im2_pts, cv2.FM_LMEDS) # Could also try CV_FM_RANSAC
    
    E , _ = cv2.findEssentialMat(im1_pts, im2_pts, cam_matrix)
    _, R, t, _ = cv2.recoverPose(E, im1_pts, im2_pts)
    H = np.hstack((R, t))
    print(R)
    print(t)
    
    print(H)
    #print(s)