
import datetime
import logging

from marvinglobal import marvinglobal as mg
from marvinglobal import skeletonCommandMethods
from marvinglobal import cartCommandMethods

processName = 'imageProcessing'

marvinShares = None   # shared data
skeletonCommandMethods = skeletonCommandMethods.SkeletonCommandMethods()   # skeleton commands
cartCommandMethods = cartCommandMethods.CartCommandMethods()

cams = {}

inForwardMove = False   # triggers monitoring with D415 cam

cartLength2 = 0.29      # meters, distance from cart center to cart front
robotWidth = 0.6        # meters
robotWidth2 = robotWidth/2        # meters
robotHeight = 1.7       # meters
robotBaseZ = 0.88       # standard table height (could be dynamic, not fully implemented yet)
robotNeckZ = 0.63       # neck rotation point above base
robotNeckY = 0.09       # neck position in relation to cart center
distOffsetCamFromCartFront = 0.1    # meters

# aruco calibration data path
calibrationData = {mg.EYE_CAM: "d:/Projekte/InMoov/imageProcessing/arucoFiles/cartcamCalibration/",
                   mg.CART_CAM: "d:/Projekte/InMoov/imageProcessing/arucoFiles/eyecamCalibration/"}

calibrationMatrix = [None] * mg.NUM_CAMS
calibrationDistortion = [None] * mg.NUM_CAMS

'''
images = {mg.EYE_CAM: None,
          mg.CART_CAM: None,
          mg.D415_RGB: None,
          mg.D415_DEPTH: None}
'''

def startLogging():
    logging.basicConfig(
        filename=f"log/{processName}.log",
        level=logging.INFO,
        format='%(message)s',
        filemode="w")


def log(msg, publish=True):

    logtime = str(datetime.datetime.now())[11:23]
    logging.info(f"{logtime} - {msg}")
    print(f"{logtime} - {msg}")

