
# limited to 32 bit python because of cart cam CL-EYE
# June 2019 added D415 sensor

import os
import sys
import time
import copy

import cv2
import imutils
import numpy as np
from cv2 import aruco
from pyrecord import Record
import glob

from marvinglobal import marvinglobal as mg
from marvinglobal import environmentClasses
from typing import List

import config


point = Record.create_type('point', 'x', 'y')

markerType = Record.create_type('markerType', 'type', 'length', 'separation', 'orthoStopDist', 'allowedIds')
# adapt length to get a correct distance
# include distance of cart front to center (330) for the orthoStopDist
markerTypes = [markerType('dockingMarker', 100, 0.5, 600, [10]),
               markerType('dockingDetail', 60, 0.5, 0, [11]),
               markerType('objectMarker', 100, 1, 600, [0,1,2])]


# ARUCO_PORT = 20001
DOCKING_MARKER_ID = 10
DOCKING_DETAIL_ID = 11

DISTANCE_CART_CENTER_CAM = 330  # mm
MARKER_XOFFSET_CORRECTION = -22

CARTCAM_X_ANGLE = 75.0
CARTCAM_X_RESOLUTION = 640

CARTCAM_Y_ANGLE = 53
CARTCAM_Y_RESOLUTION = 480

EYECAM_X_ANGLE = 70.0
EYECAM_X_RESOLUTION = 640

EYECAM_Y_ANGLE =43
EYECAM_Y_RESOLUTION = 480

# calibrateCamera
# createMarkers()
# calibration.createCharucoBoard()
# calibration.takeCalibrationPictures()
# calibration.calibrateCamera()

# calibration.calibrateCameraCharuco()
# exit()


arucoParams = aruco.DetectorParameters_create()
# this parameter creates a border around each each cell in the cell grid
arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.25     # default = 0.13
arucoParams.minDistanceToBorder = 2

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

navManager = None
cap = None
lastImg = None


def log(msg):
    if navManager is not None:
        navManager.root.recordLog("aruco - " + msg)
    print(msg)


def lookForMarkers(camType:mg.CamTypes, markerIds:List[int], camYaw, cartLocation:mg.Location):

    foundMarkers = []

    img = config.camImages[camType].image

    # check for marker
    foundIds, corners = findMarkers(img, show=False)

    if len(markerIds) > 0 and foundIds is None:
        config.log(f"none of the requested markers found in image")
        return []

    if foundIds is not None:
        #config.log(f"markers found: {foundIds}")
        for markerIndex, foundId in enumerate(foundIds):

            # check for markerId is requested
            if len(markerIds) == 0 or foundId in markerIds:
                try:
                    marker:environmentClasses.Marker = environmentClasses.Marker()
                    marker.markerId = foundId
                    marker.camType = camType
                    marker.cartLocation = cartLocation
                    marker.camLocation = copy.copy(mg.camLocation[camType])     # cam location in relation to cart center
                    marker.camLocation.addLocation(cartLocation)
                    marker.camLocation.yaw = camYaw

                    calculateMarkerFacts(corners[markerIndex], marker)

                except Exception as e:
                    config.log(f"error in calculateMarkerFacts: {e}")
                    marker = None

                if marker is not None:
                    config.log(f"markerId: {marker.markerId}, distance: {marker.distanceCamToMarker}, angleInImage: {marker.angleInImage}, "
                               f"markerYaw: {marker.markerYaw}")
                    foundMarkers.append(marker)

    return foundMarkers


def arucoTerminate():
    log("stopping arucoServer")
    print("termination request received")
    time.sleep(2)
    raise SystemExit()


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([y, x, z])


# calculate a cartYaw for cart to be positioned <distance> in front of the marker
def evalDegreesDistToCartTarget(degreesToMarker, distance, markerYaw):
    log(
        f"evalDegreesDistToCartDestination degreesToMarker: {degreesToMarker:.0f}, distance: {distance:.0f}, markerYaw: {markerYaw:.0f}")

    # Position x,y of camera
    p1 = point(0, 0)

    # Center of Marker
    p2 = point(0, 0)
    p2.x = int(distance * np.cos(np.radians(degreesToMarker)))
    p2.y = int(distance * np.sin(np.radians(degreesToMarker)))
    # print(p2)

    # angle between marker and cart destination point (includes markerYaw)
    beta = degreesToMarker - 90 + markerYaw

    # cart destination point orthogonal in front of marker with offset
    p3 = point(0, 0)
    p3.x = int(p2.x + (distance * np.sin(np.radians(beta))))
    p3.y = int(p2.y - (distance * np.cos(np.radians(beta))))
    # print(p3)

    # angle to cart point in relation to degreesToMarker
    degreesToCartTarget = 180 + np.degrees(np.arctan(p3.y / p3.x))
    distToCartTarget = np.sqrt(np.power(p3.x, 2) + np.power(p3.y, 2))

    log(
        f"degreesToCartTarget: {degreesToCartTarget:.0f}, distToCartTarget {distToCartTarget:.0f}, markerPos: {p2}, cartTargetAngle: {beta:.0f}, cartTargetPos: {p3}")

    return degreesToCartTarget, distToCartTarget


def getAvgHue(img):
    # convert to hsv for color comparison
    imgH = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avgHsvByCol = np.average(imgH, axis=0)
    avgHsv = np.average(avgHsvByCol, axis=0)
    return avgHsv[0]  # hue


# search for marker in frame
def findMarkers(cam, show):

    bwImg = cv2.cvtColor(config.cams[cam].image, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requires gray image

    # timestr = datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")
    # cv2.imwrite("images/" + timestr + ".jpg", img)

    if show:
        cv2.imshow('image for aruco check', bwImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ids = None
    corners = None
    try:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(bwImg, aruco_dict, parameters=arucoParams)  # Detect aruco
    except Exception as e:
        config.log(f"exception in detectMarkers: {e}")

    return ids, corners


def calculateMarkerFacts(corners, marker):
    """
    aruco.estimatePoseSingleMarkers looks to be a bit unprecise, use own calcs for distance and direction
    :param corners:
    :param marker:  partially filled marker object
    """

    # check for marker is an allowed marker type
    markerType = [m for m in markerTypes if marker.markerId in m.allowedIds]
    #config.log(f"markerId: {markerId}, markerSize: {marker[0].length}")

    if markerType is None or len(markerType) == 0:
        return None

    camProps = mg.camProperties[marker.camType]
    #camMatrix = config.calibrationMatrix[marker.camType]
    #camDistortionCoeffs = config.calibrationDistortion[marker.camType]

    colAngle = camProps['fovH'] / camProps['cols']
    rowAngle = camProps['fovV'] / camProps['rows']
    imgXCenter = camProps['cols'] / 2
    # use corners to calc yaw of marker
    #vec = aruco.estimatePoseSingleMarkers(corners, marker[0].length, camMatrix,
    #                                  camDistortionCoeffs)  # For a single marker

    # my markers are not rotated, use average of marker height on left and right side for distance calculation
    # corner indices are clockwise starting with topleft (second index)
    config.log(f"corners: {corners[0]}", publish=False)
    tl, tr, br, bl = 0, 1, 2, 3
    col, row = 0, 1
    centerCol = (corners[0][tr][col] + corners[0][tl][col]) / 2
    # eval left and right marker height in rows
    markerRowsLeft = corners[0][bl][row] - corners[0][tl][row]
    markerRowsRight = corners[0][br][row] - corners[0][tr][row]
    markerRows = (markerRowsLeft + markerRowsRight) / 2

    # eval the angle of the marker in the image using the vertical cam angle and resolution
    heightAngle = markerRows * rowAngle

    # using the known size of the marker the distance is adj=opp/tan
    # use abs value as eye cam delivers a rotated map
    marker.distanceCamToMarker = abs(marker[0].length / np.tan(np.radians(heightAngle)))

    # use the markers center col to calc the angle to the marker
    marker.angleInImage = (imgXCenter - centerCol) * colAngle
    config.log(f"angleInImage, centerCol: {centerCol}, offset: {imgXCenter - centerCol}, colAngle: {colAngle}")

    # eval the marker's yaw from the result of aruco.estimatePoseSingleMarkers
    rmat = cv2.Rodrigues(vec[0])[0]
    yrp = rotationMatrixToEulerAngles(rmat)

    # markerYaw is 0 for an orthogonal view position,
    # negative for a viewpoint right of the marker
    # positive for a viewpoint left of the marker
    marker.markerLocation.yaw = float(-np.degrees(yrp[0]))  # ATTENTION, this is the yaw of the marker evaluated from the image

    # for distances > 1500 markerYaw are not accurate, reduce value
    if marker.distanceCamToMarker > 1500:
        config.log(f"corrected markerYaw from {marker.markerLocation.yaw} to {marker.markerLocation.yaw/3} because of distance {marker.distanceCamToMarker}")
        marker.markerLocation.yaw = round(marker.markerLocation.yaw / 3)

    log(f"markerId: {marker.markerId}, distanceCamToMarker: {marker.distanceCamToMarker:.0f}, angleInImage: {marker.angleInImage:.2f}, markerYaw: {marker.markerLocation.yaw:.2f}")



def calculateCartMovesForDockingPhase1(corners):
    cartCenterPos = point(0, 0)
    camPos = point(0, DISTANCE_CART_CENTER_CAM)

    marker = markerTypes[markerType == "dockingMarker"]

    vec = aruco.estimatePoseSingleMarkers(corners, marker.length, config.cartcamMatrix,
                                          config.cartcamDistortionCoeffs)  # For a single marker

    distanceCamToMarker = vec[1][0, 0, 2]
    xOffsetMarker = vec[1][0, 0, 0]
    log(f"distanceCamToMarker: {distanceCamToMarker:.0f}, xOffsetMarker: {xOffsetMarker:.0f}")
    # log(f"vec[1] {vec[1]}")

    # angle of marker center in cam image, atan of x-offset/distanceCamToMarker
    markerAngleInImage = np.degrees(np.arctan(-xOffsetMarker / distanceCamToMarker))

    # calculate marker position relativ to cart center (cartcam looks straight out)
    markerCenterPos = point(xOffsetMarker, camPos.y + np.cos(np.radians(markerAngleInImage)) * distanceCamToMarker)
    log(
        f"markerAngleInImage: {markerAngleInImage:.1f}, markerCenterPos(relative to cart center): {markerCenterPos.x:.0f} / {markerCenterPos.y:.0f}")

    # eval the marker's yaw (the yaw of the marker itself evaluated from the marker corners)
    rmat = cv2.Rodrigues(vec[0])[0]
    yrp = rotationMatrixToEulerAngles(rmat)
    markerYaw = -np.degrees(yrp[0])  # ATTENTION, this is the yaw of the marker evaluated from the image

    # orthoAngle = markerAngleInImage + markerYaw
    orthoAngle = markerYaw
    xCorr = (marker.orthoStopDist) * np.sin(np.radians(orthoAngle))
    yCorr = (marker.orthoStopDist) * np.cos(np.radians(orthoAngle))
    log(
        f"markerYaw: {markerYaw:.1f}, orthoAngle: {orthoAngle:.1f}, xCorr: {xCorr:.0f}, yCorr: {yCorr:.0f}")

    # cart target position is marker's orthogonal point at distance
    # use the offset to account for the x-difference of the docking detail marker center vs the docking marker center
    cartTargetPos = point(markerCenterPos.x + xCorr + MARKER_XOFFSET_CORRECTION, markerCenterPos.y - yCorr)

    log(f"cartTargetPos (cartCenter) = {cartTargetPos.x:.0f},{cartTargetPos.y:.0f}")

    cartStartRotation = np.degrees(np.arctan(cartTargetPos.x / cartTargetPos.y))
    cartMove = np.hypot(cartTargetPos.x, cartTargetPos.y)
    cartEndRotation = -(np.degrees(np.arctan(xCorr / yCorr)) + cartStartRotation)

    return [cartStartRotation, cartMove, cartEndRotation]


def calculateCartMovesForDockingPhase2(corners):
    '''
    cart is expected to be in front of the docking station
    calculate detail moves with the DOCKING_DETAIL_MARKER
    '''

    marker = markerTypes[markerType == "dockingDetail"]

    vec = aruco.estimatePoseSingleMarkers(corners, marker.length, config.cartcamMatrix,
                                          config.cartcamDistortionCoeffs)  # For a single marker

    distanceCamToMarker = vec[1][0, 0, 2]
    xOffsetMarker = vec[1][0, 0, 0]

    # eval the marker's yaw (the yaw of the marker itself evaluated from the marker corners)
    rmat = cv2.Rodrigues(vec[0])[0]
    yrp = rotationMatrixToEulerAngles(rmat)
    markerYaw = np.degrees(yrp[0])  # ATTENTION, this is the yaw of the marker evaluated from the image

    # rotate only if we are not orthogonal to the marker
    rotation = 0
    if abs(markerYaw) > 2:
        rotation = markerYaw

    log(f"rotation: {rotation:.0f}, xOffsetMarker: {xOffsetMarker:.0f}, distanceCamToMarker: {distanceCamToMarker:.0f}")

    return rotation, xOffsetMarker, distanceCamToMarker


def initAruco():

    createMarkers = False
    if createMarkers:
        createMarkers()
        raise SystemExit(0)

    takeCalibrationPicturesEyecam = False
    if takeCalibrationPicturesEyecam:
        path = config.arucoWithEyeCam
        camID = 0
        time.sleep(10)
        takeCalibrationPictures(path, camID, -90)
        createCalibrationMatrixFromImages(path)
        raise SystemExit(0)

    takeCalibrationPicturesCartcam = False
    if takeCalibrationPicturesCartcam:
        path = "/home/marvin/InMoov/aruco/cartcamCalibration/"
        camID = 1
        #calibration.takeCalibrationPictures(path, camID, 0)
        createCalibrationMatrixFromImages(path)
        raise SystemExit(0)


    data = np.load("/home/marvin/InMoov/imageProcessing/arucoFiles/cartcamCalibration/calibration.npz")
    config.cartcamMatrix = data['cameraMatrix']
    config.cartcamDistortionCoeffs = data['distortionCoeffs'][0]

    data = np.load("/home/marvin/InMoov/imageProcessing/arucoFiles/eyecamCalibration/calibration.npz")
    config.calibrationMatrix[mg.CamTypes.EYE_CAM.value] = data['cameraMatrix']
    config.calibrationDistortion[mg.CamTypes.EYE_CAM.value] = data['distortionCoeffs'][0]


def createMarkers():
    for i in range(0, 20):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
        img = aruco.drawMarker(aruco_dict, i, 500)
        cv2.imwrite("marker" + str(i) + ".jpg", img)


def createCharucoBoard():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
    board = aruco.CharucoBoard_create(6, 8, 0.025, 0.02, aruco_dict)
    img = board.draw((200 * 3, 200 * 3))
    cv2.imwrite("CharucoBoard.jpg", img)


def takeCalibrationPictures(path, cam, rotation):
    arucoParams = aruco.DetectorParameters_create()
    arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_50)
    board = aruco.CharucoBoard_create(6, 8, 0.025, 0.02, arucoDict)

    cap = cv2.VideoCapture(cam)
    time.sleep(1)
    if not cap.isOpened():
        cap.open()

    i = 0
    while i < 30:

        config.cams[cam].takeImage()

        rows, cols = img.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

        cv2.imshow('charuco', img)
        cv2.waitKey(1000)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, corners = cv2.findChessboardCorners(gray, (5,6),None)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict,
                                                              parameters=arucoParams)  # Detect aruco
        # aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedImgPoints)

        # print("# ids: " + str(len(ids)))
        if ids is not None:
            i += 1
            filename = path + "calibrationImage_" + str(i) + ".jpg"
            cv2.imwrite(filename, img)
            print("image " + filename + " saved")
        else:
            print(ids)


def createCalibrationMatrixFromImages(path):
    arucoParams = aruco.DetectorParameters_create()
    arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_50)
    board = aruco.CharucoBoard_create(6, 8, 0.025, 0.02, arucoDict)

    allCorners = []
    allIds = []

    images = glob.glob(path + '*.jpg')
    imsize = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imsize = gray.shape

        res = cv2.aruco.detectMarkers(gray, arucoDict)

        if len(res[0]) > 0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3:
                allCorners.append(res2[1])
                allIds.append(res2[2])

            cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    try:
        cal = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, imsize, None, None)
        np.savez(path + "calibration", cameraMatrix=cal[1], distortionCoeffs=cal[2])
        print(f"calibration data saved, path: {path}")
        print(cal[1])
        print(cal[2])
    except:
        print("no calibration possible with provided images")

    cv2.destroyAllWindows()



if __name__ == "__main__":


    config.log(f"{config.processName} started")


    windowName = f"marvin//{config.processName}"
    os.system("title " + windowName)

    standAloneMode = False
    if standAloneMode:
        import imageProcessing

        imageProcessing.initCameras()

        #arucoParams.polygonalApproxAccuracyRate = 0.08
        arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.15     # default = 0.13
        while True:

            file = False
            if file:
                filename = "C:/Users/marvin/Desktop/000_000_300_-30.jpg"
                config.cams[mg.CamTypes.EYE_CAM].loadImage(filename)
                config.cams[mg.CamTypes.CART_CAM].loadImages(filename)

            else:
                config.cams[mg.CamTypes.EYE_CAM].takeImage()
                config.cams[mg.CamTypes.CART_CAM].takeImage()

            # CARTCAM
            if config.cams[mg.CamTypes.CART_CAM].image is not None:
                cv2.imshow("cartCam", config.cams[mg.CamTypes.CART_CAM].image)

                ids, corners = findMarkers(mg.CamTypes.CART_CAM, False)
                if ids is not None:
                    print(ids)
                    for markerIndex, foundId in enumerate(ids):
                        markerInfo = calculateMarkerFacts(corners[markerIndex], foundId, "CART_CAM")
                        #config.log(f"corners: {corners}")
                        config.log(f"markerInfo: {markerInfo}")
                else:
                    log(f"no marker found")

                cv2.waitKey(0)


            # EYECAM
            if config.cams[mg.CamTypes.EYE_CAM].image is not None:
                cv2.imshow("eyeCam", config.cams[mg.CamTypes.EYE_CAM].image)

                ids, corners = findMarkers(mg.CamTypes.EYE_CAM, False)

                if ids is not None:
                    print(ids)
                    for markerIndex, foundId in enumerate(ids):
                        markerInfo = calculateMarkerFacts(corners[markerIndex], foundId, "EYE_CAM")
                else:
                    log(f"no marker found")

                cv2.waitKey(0)
                cv2.destroyAllWindows()

            raise SystemExit(0)
