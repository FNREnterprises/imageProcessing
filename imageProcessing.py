

# currently needs python version 3.7 because of pyrealsense2

import os
import threading
import queue
from dataclasses import dataclass

from marvinglobal import marvinglobal as mg
from marvinglobal import marvinShares
from marvinglobal import servoCommandMethods
from marvinglobal import cartCommandMethods

import config
import aruco
import camClasses
import monitorForwardMove

def initCameras():
    '''
    def __init__(self, name, deviceId, cols, rows, fovH, fovV, rotate, numReads):
    :return:
    '''
    viewInitialImages = 0   # set to 2000 to get a window iwth tie image
    config.cams.update({mg.CART_CAM: camClasses.UsbCamera("cartcam", 3, 640, 480, 60, 60, 0, 2)})
    if config.cams[mg.CART_CAM].takeImage(viewInitialImages):
        config.cams[mg.CART_CAM].saveImage()

    config.cams.update({mg.EYE_CAM: camClasses.UsbCamera("eyecam", 1, 640, 480, 21, 40, -90, 2)})
    if config.cams[mg.EYE_CAM].takeImage(viewInitialImages):
        config.cams[mg.EYE_CAM].saveImage()

    config.cams.update({mg.D415: camClasses.D415Camera("headcam", None, 1920, 1080, 69.4, 42.5, 0, 5)})
    if config.cams[mg.D415].takeImage(viewInitialImages):
        config.cams[mg.D415].saveImage()
    if config.cams[mg.D415].takeDepth(viewInitialImages):
        config.cams[mg.D415].saveDepth()

    #config.cams.update({mg.D415_DEPTH: camClasses.D415Camera("headcam", None, 1920, 1080, 69.4, 42.5, 0, 5)})
    #config.cams[mg.D415_DEPTH].takeDepth(2000)


@dataclass
class CheckImageForArucoCode:
    def __init__(self, cam, markers, file):
        self.requestType = mg.CHECK_FOR_ARUCO_CODE
        self.cam = cam
        self.markerList = markers
        self.imageFile = file


if __name__ == '__main__':

    config.startLogging()

    initCameras()

    config.log(f"cams: {config.cams.keys()}")

    config.marvinShares = marvinShares.MarvinShares()
    if not config.marvinShares.sharedDataConnect(config.processName):
        config.log(f"could not connect with marvinData")
        os._exit(1)

    #config.servoCommandMethods = servoCommandMethods.ServoCommandMethods()
    #config.cartCommandMethods = cartCommandMethods.CartCommandMethods()

    # prepare for aruco markers
    aruco.initAruco()

    # standalone-options for testing
    testing = True
    if testing:
        aruco = False
        if aruco:
            fromFile = False
            if fromFile:
                config.marvinShares.imageProcessingQueue.put([mg.CHECK_FOR_ARUCO_CODE, mg.EYE_CAM, [], "d:/projekte/inmoov/imageProcessing/arucoFiles/eyecam2.jpg"])
            else:
                config.marvinShares.imageProcessingQueue.put([mg.CHECK_FOR_ARUCO_CODE, mg.EYE_CAM, []])

        simulateForwardMove = True
        if simulateForwardMove:
            config.marvinShares.imageProcessingQueue.put([mg.START_MONITOR_FORWARD_MOVE])


    config.log(f"start forward move monitoring thread")
    monitorThread = threading.Thread(target=monitorForwardMove.monitorLoop, args={})
    monitorThread.setName("monitorThread")
    monitorThread.start()

    config.log(f"waiting for imageProcessing commands")

    while True:

        try:
            config.marvinShares.updateProcessDict(config.processName)
            request = config.marvinShares.imageProcessingQueue.get(block=True, timeout=1)

        except queue.Empty: # in case of empty queue update processDict only
            continue

        except TimeoutError: # in case of timeout update processDict only
            continue

        except Exception as e:
            config.log(f"exception in waiting for imageProcessing requests, {e}, going down")
            os._exit(1)

        if request[0] == mg.CHECK_FOR_ARUCO_CODE:
            # [<requestType>,<cam-id>,<markerlist(empty list for any)>,optional<imageFile>]
            cam = request[1]
            markers = request[2]
            if len(request) == 4:
                # request contains a filepath to read image from
                file = request[3]
                config.cams[cam].loadImage(file)

            else:
                config.cams[cam].takeImage(0)

            foundMarkers = aruco.findMarkers(cam, True)

            config.log(f"foundMarkers={foundMarkers}")


        if request[0] == mg.START_MONITOR_FORWARD_MOVE:

            # activate monitoring ground, woll and ahead space
            config.inForwardMove = True


        if request[0] == mg.STOP_MONITOR_FORWARD_MOVE:
            config.inForwardMove = False

