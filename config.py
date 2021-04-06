
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
camImages = {}

inForwardMove = False   # triggers monitoring with D415 cam


# aruco calibration data path
calibrationData = {mg.CamTypes.EYE_CAM: f"{mg.INMOOV_BASE_FOLDER}/imageProcessing/arucoFiles/cartcamCalibration/",
                   mg.CamTypes.CART_CAM: f"{mg.INMOOV_BASE_FOLDER}/imageProcessing/arucoFiles/eyecamCalibration/"}

calibrationMatrix = [None] * mg.NUM_CAMS
calibrationDistortion = [None] * mg.NUM_CAMS

'''
images = {mg.CamTypes.EYE_CAM: None,
          mg.CamTypes.CART_CAM: None,
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

