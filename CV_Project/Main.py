import HelperFunc
import cv2
import os.path
if __name__ == '__main__':
    # NOTE: Only run camera calibration when need to re-calibrate camera
    #       This function will take a long time to run for very large image
    
    #HelperFunc.calibrateCamera()
    cam_matrix, distortion = HelperFunc.loadCameraCalibration()
    currDir = os.getcwd()
    im1 = cv2.imread(os.path.join(currDir, 
                                  "Images\\Stereo Images\\0in.JPG"))
    im2 = cv2.imread("Images\\Stereo Images\\6in.JPG")
    HelperFunc.findMatchingPoints(im1, im2)