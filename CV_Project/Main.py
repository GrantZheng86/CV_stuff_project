import HelperFunc
import cv2
import os.path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # NOTE: Only run camera calibration when need to re-calibrate camera
    #       This function will take a long time to run for very large image

    # HelperFunc.calibrateCamera()

    # NOTE: Do not try to undistort the camera, peraxial images can be solved very well,
    #       but does not work good for other cases.
    cam_matrix, distortion = HelperFunc.loadCameraCalibration()
    currDir = os.getcwd()
    im1 = cv2.imread(os.path.join(currDir,
                                  "Images\\Stereo Images\\36in.JPG"))
    im2 = cv2.imread("Images\\Stereo Images\\42in.JPG")

    #     im1 = cv2.resize(im1, (0, 0), fx = 0.2, fy=0.2 )
    #     im2 = cv2.resize(im2, (0, 0), fx = 0.2, fy=0.2 )

    im1_pts, im2_pts = HelperFunc.findMatchingPoints(im1, im2, True)
    im1_pts = np.float32(im1_pts)
    im2_pts = np.float32(im2_pts)

    # Finding the fundamental matrix
    F, mask = cv2.findFundamentalMat(im1_pts, im2_pts, cv2.FM_LMEDS)  # Could also try CV_FM_RANSAC

    E, _ = cv2.findEssentialMat(im1_pts, im2_pts, cam_matrix)
    _, R, t, _ = cv2.recoverPose(E, im1_pts, im2_pts)
    H = np.hstack((R, t))
    print('rotation:\n',R)
    print('translation:\n',t)

    print('homographys:\n',H)
    #
    # calculating the scale to the real world. The translation of the camera is known
    # to be 6 inches, 152.4mm
    scale = 152.4 / np.abs(t[0])
    print('scale: ',scale)
    p_reference = np.matmul(cam_matrix, np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1))
    p_second = np.matmul(cam_matrix, H)
    # make sure that p_second(3,3) is >= 0 since all the points are infront of both
    # camera positions
    print(p_second)

    # use triangulation from user indicated points
    im1 = cv2.resize(im1, (0, 0), fx=0.2, fy=0.2)
    im2 = cv2.resize(im2, (0, 0), fx=0.2, fy=0.2)
    # select points in a clockwise fashion beginning at the upper left
    clickedPoints = HelperFunc.choosePointsOnImage(im1)
    clickedPoints = HelperFunc.choosePointsOnImage(im2)
    clickedPoints = np.array(clickedPoints)
    clickedPoints = np.multiply(clickedPoints, 5)
    im1_pts = clickedPoints[0:4, :]
    im2_pts = clickedPoints[4:8, :]
    points = []
    for i in range(len(im1_pts)):
        A = [[im2_pts[i, 0] * p_second[2, :] - p_second[0, :]],
             [im2_pts[i, 1] * p_second[2, :] - p_second[1, :]],
             [im1_pts[i, 0] * p_reference[2, :] - p_reference[0, :]],
             [im1_pts[i, 1] * p_reference[2, :] - p_reference[1, :]]]
        A = np.squeeze(np.array(A))
        _, _, v = np.linalg.svd(A)
        v = np.transpose(v)
        curr = v[:, 3]
        curr = curr / curr[3]
        points.append(curr)

    print(f"The vectors from the camera to the selected keypoints (in milimeters) are:  ")
    vectors = []
    for each_point in points:
        x = each_point[0]
        y = each_point[1]
        z = each_point[2]
        corners = [x, y, z] * scale
        print(corners)
        vectors.append(corners)

    # x length between object points
    reconstructed_x = abs(abs(vectors[0][0]) - abs(vectors[1][0]))
    # y length between object points
    reconstructed_y = abs(abs(vectors[1][1])-abs(vectors[2][1]))
    print(f'object is: {reconstructed_x}mm by {reconstructed_y}mm')
    plt.show()
         
        
        
        
