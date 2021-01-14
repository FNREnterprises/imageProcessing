
import time
import processDepth

import config
import findObstacles

from marvinglobal import marvinglobal as mg
from marvinglobal import marvinShares
from marvinglobal import servoCommandMethods

# activated through imageProcessingQueue
# checks for ground and wall obstacles

def setRotheadNeck(rothead, neck):
    '''
    move head and neck, verify with headImu and return new yaw,pitch
    return yaw, pitch
    '''
    q = config.marvinShares.servoRequestQueue
    m = config.servoCommandMethods
    m.positionServo(q, "head.rothead", rothead, 500)
    m.positionServo(q, "head.neck", neck, 500)
    time.sleep(1)
    headImu = config.marvinShares.cartDict.get(mg.SharedDataItem.HEAD_IMU)
    return headImu.yaw, headImu.pitch


def monitorLoop():

    checkGround = False
    checkAhead = False
    checkWall = False

    config.log(f"start obstacle monitoring during forward move")
    d415 = config.cams[mg.D415]

    while True:

        if not config.inForwardMove:
            checkGround = False     # check Ground is done first time in move request
            checkAhead = True       # when cart starts to move check for ahead obstacles (0.5..2m)
            checkWall = False       # check wall follows after checkAhead
            if d415.isCamStreaming():
                d415.stopStream()
            time.sleep(1)
            continue

        if not d415.isCamStreaming():
            if not d415.startStream():
                config.log(f"error in restarting D415 image stream")
                # stop cart

        if checkGround:

            config.log(f"D415 ground check")

            # set headcam into watch ground position
            #def positionServo(requestQueue, servoName, position, duration):

            setRotheadNeck(0, config.pitchGroundWatchDegrees)

            if not d415.takeDepth():
                config.log(f"could not acquire depth points")
                return False

            distClosestObstacle, obstacleDirection = findObstacles.distGroundObstacles(d415.depth)

            if distClosestObstacle > 0.2:
                config.log(f"continue driving, free move: {distClosestObstacle * 100:.0f} cm")
                checkAhead = True   # next check is for ahead obstacles
                checkGround = False

            else:
                config.log(
                    f"stop move forward because of a ground obstacle ahead in {distClosestObstacle * 100:.0f} cm at {obstacleDirection:.0f} degrees")
                config.cartCommandMethods.stopCart(config.marvinShares.cartRequestQueue, "ground object detected")
                config.flagInForwardMove = False
                d415.stopStream()


        if checkAhead:   # check medium distance

            config.log(f"depth ahead check")

            yaw, pitch = setRotheadNeck(0, config.pitchAheadWatchDegrees)
            freeMoveDistance = 0

            if not d415.takeDepth():
                config.log(f"could not acquire the depth points")

            else:
                points = processDepth.alignPointsWithGround(d415.depth, config.pitchAheadWatchDegrees)

                freeMoveDistance, obstacleDirection = findObstacles.lookForCartPathObstacle(points)
                config.log(f"obstacle in path: freeMoveDistance: {freeMoveDistance:.3f}, obstacleDirection: {obstacleDirection:.1f}")


            if freeMoveDistance > 0.4:
                config.log(f"continue driving, free move: {freeMoveDistance * 100:.0f} cm")
                checkAhead = False
                checkWall = True        # next check is for wall


            else:
                config.log(
                    f"stop move forward because of an obstacle ahead in {freeMoveDistance * 100:.0f} cm at {obstacleDirection:.0f} degrees")
                config.cartCommandMethods.stopCart(config.marvinShares.cartRequestQueue,"object in cart path detected")
                config.flagInForwardMove = False


        if checkWall:   # check wall

            config.log(f"depth wall check")

            yaw, pitch = setRotheadNeck(0, config.pitchWallWatchDegrees)
            if yaw is None:
                config.log(f"moveRequest, could not position head")
                return False

            points = camImages.cams[inmoovGlobal.HEAD_CAM].takeDepth()

            if points is None:
                config.log(f"could not acquire depth points")
                return None, None

            distClosestObstacle, obstacleDirection, _ = findObstacles.aboveGroundObstacles(points, inmoovGlobal.pitchWallWatchDegrees)

            if distClosestObstacle > 0.4:
                config.log(f"continue driving, free move: {distClosestObstacle * 100:.0f} cm")
                checkGround = True      # revert back to ground check
                checkWall = False

            else:
                config.log(
                    f"stop move forward because of a wall obstacle ahead in {distClosestObstacle * 100:.0f} cm at {obstacleDirection:.0f} degrees")
                arduinoSend.sendStopCommand("wall object detected")
                config.flagInForwardMove = False
                depthImage.stopD415Stream()
