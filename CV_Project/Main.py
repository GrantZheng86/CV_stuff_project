
import HelperFunc

if __name__ == '__main__':
    # NOTE: Only run camera calibration when need to re-calibrate camera
    #       This function will take a long time to run for very large image
    HelperFunc.calibrateCamera()
    a, b = HelperFunc.loadCameraCalibration()
    print(a)