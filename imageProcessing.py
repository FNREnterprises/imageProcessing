

# currently needs python version 3.7 because of pyrealsense2

import os
import threading
import queue
from dataclasses import dataclass
import cv2

from marvinglobal import marvinglobal as mg
from marvinglobal import marvinShares

import config
import aruco
import camClasses
import monitorForwardMove


def initCameras():
    '''
    def __init__(self, name, deviceId, cols, rows, fovH, fovV, rotate, numReads):
    :return:
    '''
    viewInitialImages = 0   # set to 2000 to get a window with the image
    config.cams.update({mg.CamTypes.CART_CAM: camClasses.UsbCamera(mg.camProperties[mg.CamTypes.CART_CAM])})
    if config.cams[mg.CamTypes.CART_CAM].takeImage(viewInitialImages):
        config.cams[mg.CamTypes.CART_CAM].saveImage("initialCamImages/{self.name}Rgb.jpg")

    config.cams.update({mg.CamTypes.EYE_CAM: camClasses.UsbCamera(mg.camProperties[mg.CamTypes.EYE_CAM])})
    if config.cams[mg.CamTypes.EYE_CAM].takeImage(viewInitialImages):
        config.cams[mg.CamTypes.EYE_CAM].saveImage("initialCamImages/{self.name}Rgb.jpg")

    config.cams.update({mg.CamTypes.HEAD_CAM: camClasses.D415Camera(mg.camProperties[mg.CamTypes.HEAD_CAM])})
    if config.cams[mg.CamTypes.HEAD_CAM].takeImage(viewInitialImages):
        config.cams[mg.CamTypes.HEAD_CAM].saveImage("initialCamImages/{self.name}Rgb.jpg")
    if config.cams[mg.CamTypes.HEAD_CAM].takeDepth(viewInitialImages):
        config.cams[mg.CamTypes.HEAD_CAM].saveDepth("initialCamImages/{self.name}Rgb.jpg")

    #config.cams.update({mg.D415_DEPTH: camClasses.D415Camera("headcam", None, 1920, 1080, 69.4, 42.5, 0, 5)})
    #config.cams[mg.D415_DEPTH].takeDepth(2000)


@dataclass
class CheckImageForArucoCode:
    def __init__(self, cam, markers, file):
        self.requestType = mg.ImageProcessingCommands.CHECK_FOR_ARUCO_CODE
        self.cam = cam
        self.markerList = markers
        self.imageFile = file


if __name__ == '__main__':

    config.startLogging()

    initCameras()

    #config.log(f"cams: {config.cams.keys()}")

    config.marvinShares = marvinShares.MarvinShares()
    if not config.marvinShares.sharedDataConnect(config.processName):
        config.log(f"could not connect with marvinData")
        os._exit(1)

    #config.SkeletonCommandsMethods = SkeletonCommandsMethods.SkeletonCommandsMethods()
    #config.CartCommandsMethods = CartCommandsMethods.CartCommandsMethods()

    # prepare for aruco markers
    aruco.initAruco()

    # standalone-options for testing
    testing = True
    if testing:
        aruco = False
        if aruco:
            fromFile = False
            if fromFile:
                config.marvinShares.imageProcessingQueue.put(config.processName, mg.ImageProcessingCommands.CHECK_FOR_ARUCO_CODE, mg.CamTypes.EYE_CAM, [], "d:/projekte/inmoov/imageProcessing/arucoFiles/eyecam2.jpg")
            else:
                config.marvinShares.imageProcessingQueue.put(config.processName, mg.ImageProcessingCommands.CHECK_FOR_ARUCO_CODE, mg.CamTypes.EYE_CAM, [])

        simulateForwardMove = False
        if simulateForwardMove:
            msg = {'sender': config.processName, 'cmd': mg.ImageProcessingCommands.START_MONITOR_FORWARD_MOVE}
            config.marvinShares.imageProcessingQueue.put(msg)


    config.log(f"start forward move monitoring thread")
    monitorThread = threading.Thread(target=monitorForwardMove.monitorLoop, args={})
    monitorThread.setName("monitorThread")
    monitorThread.start()

    config.log(f"waiting for imageProcessing commands")

    while True:

        request = {'cmd': None}
        try:
            config.marvinShares.updateProcessDict(config.processName)
            request:dict = config.marvinShares.imageProcessingQueue.get(block=True, timeout=1)

        except queue.Empty: # in case of empty queue update processDict only
            continue

        except TimeoutError: # in case of timeout update processDict only
            continue

        except Exception as e:
            config.log(f"exception in waiting for imageProcessing requests, {e}, going down")
            os._exit(1)


        if request['cmd'] == mg.ImageProcessingCommands.TAKE_IMAGE:
            cam = request['cam']            # e.g. mg.CamTypes.EYE_CAM
            if not config.cams[cam].takeImage(0):
                if request['sender'] == 'navManager':
                    response = {'cmd': mg.NavManagerCommands.TAKE_IMAGE_RESULT, 'sender': config.processName,
                        'success': False}
                    config.marvinShares.navManagerRequestQueue.put(response)

            if 'show' in request and request['show'] == True:
                cv2.imshow(f"{cam}", config.cams[cam].image)
                cv2.waitKey(request['showDuration'])
                cv2.destroyAllWindows()

            if 'imgPath' in request:
                cv2.imwrite(request['imgPath'], config.cams[cam].image)
                config.log(f"image saved: {cam}, {request['imgPath']}")

            if 'findMarkers' in request:
                cartLocation = config.marvinShares.cartDict.get(mg.SharedDataItems.CART_LOCATION)
                if cam == mg.CamTypes.EYE_CAM:
                    camYaw = config.marvinShares.servoCurrentDict.get('head.neck').degrees
                else:
                    camYaw = 0
                foundMarkers = aruco.lookForMarkers(cam, request['findMarkers'], camYaw, cartLocation, show=True)
                config.log(f"foundMarkers={foundMarkers}")
                response = {'cmd': mg.NavManagerCommands.FOUND_MARKERS, 'sender': config.processName,
                            'markers': foundMarkers}
                config.marvinShares.navManagerRequestQueue.put(response)


        if request['cmd'] == mg.ImageProcessingCommands.CHECK_FOR_ARUCO_CODE:
            # [<requestType>,<cam-id>,<markerlist(empty list for any)>,optional<imageFile>]
            cam = request['cam']
            markers = request['markers']
            if request['inFile'] is not None:
                # request contains a filepath to read image from
                config.log(f"check for aruco codes {markers} in {request['inFile']}")
                config.cams[cam].loadImage(request['inFile'])

            else:
                config.log(f"check for aruco codes {markers} in {config.cams[cam]['name']} image ")
                config.cams[cam].takeImage(0)
                if request['outFile'] is not None:
                    config.log(f"save image in path {request['outFile']}")
                    config.cams[cam].saveImage(request['outFile'])


            foundMarkers = aruco.findMarkers(cam, True)


        if request['cmd'] == mg.ImageProcessingCommands.START_MONITOR_FORWARD_MOVE:

            config.log(f"start obstacle scanning with head cam in forward move")
            # check for running skeletonControl
            if "skeletonControl" not in config.marvinShares.processDict.keys():
                config.log(f"skeletonControl not running, imageProcessing needs to control head settings")
                config.log(f"START_MONITOR_FORWARD_MOVE ignored")
                continue

            if "cartControl" not in config.marvinShares.processDict.keys():
                config.log(f"cartControl not running, imageProcessing needs to read headImu")
                config.log(f"START_MONITOR_FORWARD_MOVE ignored")
                continue

            # activate monitoring ground, woll and ahead space
            config.inForwardMove = True


        if request['cmd'] == mg.ImageProcessingCommands.STOP_MONITOR_FORWARD_MOVE:
            config.log(f"stop obstacle scanning with head cam")
            config.inForwardMove = False

