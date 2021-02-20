

# currently needs python version 3.7 because of pyrealsense2

import os
import threading
import queue
from dataclasses import dataclass

from marvinglobal import marvinglobal as mg
from marvinglobal import marvinShares
from marvinglobal import skeletonCommandMethods
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
    config.cams.update({mg.CART_CAM: camClasses.UsbCamera(mg.cartCamProperties)})
    if config.cams[mg.CART_CAM].takeImage(viewInitialImages):
        config.cams[mg.CART_CAM].saveImage("initialCamImages/{self.name}Rgb.jpg")

    config.cams.update({mg.EYE_CAM: camClasses.UsbCamera(mg.eyeCamProperties)})
    if config.cams[mg.EYE_CAM].takeImage(viewInitialImages):
        config.cams[mg.EYE_CAM].saveImage("initialCamImages/{self.name}Rgb.jpg")

    config.cams.update({mg.D415: camClasses.D415Camera(mg.D415CamProperties)})
    if config.cams[mg.D415].takeImage(viewInitialImages):
        config.cams[mg.D415].saveImage("initialCamImages/{self.name}Rgb.jpg")
    if config.cams[mg.D415].takeDepth(viewInitialImages):
        config.cams[mg.D415].saveDepth("initialCamImages/{self.name}Rgb.jpg")

    #config.cams.update({mg.D415_DEPTH: camClasses.D415Camera("headcam", None, 1920, 1080, 69.4, 42.5, 0, 5)})
    #config.cams[mg.D415_DEPTH].takeDepth(2000)


@dataclass
class CheckImageForArucoCode:
    def __init__(self, cam, markers, file):
        self.requestType = mg.ImageProcessingCommand.CHECK_FOR_ARUCO_CODE
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

    #config.skeletonCommandMethods = skeletonCommandMethods.skeletonCommandMethods()
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
                config.marvinShares.imageProcessingQueue.put(config.processName, mg.ImageProcessingCommand.CHECK_FOR_ARUCO_CODE, mg.EYE_CAM, [], "d:/projekte/inmoov/imageProcessing/arucoFiles/eyecam2.jpg")
            else:
                config.marvinShares.imageProcessingQueue.put(config.processName, mg.ImageProcessingCommand.CHECK_FOR_ARUCO_CODE, mg.EYE_CAM, [])

        simulateForwardMove = False
        if simulateForwardMove:
            msg = {'sender': config.processName, 'cmd': mg.ImageProcessingCommand.START_MONITOR_FORWARD_MOVE}
            config.marvinShares.imageProcessingQueue.put(msg)


    config.log(f"start forward move monitoring thread")
    monitorThread = threading.Thread(target=monitorForwardMove.monitorLoop, args={})
    monitorThread.setName("monitorThread")
    monitorThread.start()

    config.log(f"waiting for imageProcessing commands")

    while True:

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

        if request['cmd'] == mg.ImageProcessingCommand.CHECK_FOR_ARUCO_CODE:
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

            config.log(f"foundMarkers={foundMarkers}")
            mg.signalCaller(request[0], mg.ImageProcessingCommand.ARUCO_CHECK_RESULT, foundMarkers)


        if request['cmd'] == mg.ImageProcessingCommand.START_MONITOR_FORWARD_MOVE:

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


        if request['cmd'] == mg.ImageProcessingCommand.STOP_MONITOR_FORWARD_MOVE:
            config.log(f"stop obstacle scanning with head cam")
            config.inForwardMove = False

