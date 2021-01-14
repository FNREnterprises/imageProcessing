
import datetime
import logging

from marvinglobal import marvinglobal as mg
from marvinglobal import servoCommandMethods
from marvinglobal import cartCommandMethods

processName = 'imageProcessing'

marvinShares = None   # shared data
servoCommandMethods = servoCommandMethods.ServoCommandMethods()   # skeleton commands
cartCommandMethods = cartCommandMethods.CartCommandMethods()

cams = {}

inForwardMove = False   # triggers monitoring with D415 cam

# ground watch position of head for depth
pitchGroundWatchDegrees = -35   # head.neck

# ahead watch position of head for depth
pitchAheadWatchDegrees = -15   # head.neck



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

