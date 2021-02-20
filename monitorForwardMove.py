
import time
import processDepth

import config
import findObstacles

from marvinglobal import marvinglobal as mg
from marvinglobal import marvinShares
from marvinglobal import skeletonCommandMethods

# activated through imageProcessingQueue
# checks for ground and wall obstacles

def setHeadDegrees(requestedRotheadDegrees, requestedNeckDegrees):
    '''
    move head and neck, verify with headImu and return new yaw,pitch
    return yaw, pitch
    '''
    q = config.marvinShares.skeletonRequestQueue
    m = config.skeletonCommandMethods
    m.requestDegrees(q, "head.rothead", requestedRotheadDegrees, 500)
    m.requestDegrees(q, "head.neck", requestedNeckDegrees, 500)

    startPositioningTime = time.time()
    while True:
        headImu = config.marvinShares.cartDict.get(mg.SharedDataItem.HEAD_IMU)
        rotheadPositioned = abs(headImu.yaw - requestedRotheadDegrees) < 3
        neckPositioned = abs(headImu.pitch - requestedNeckDegrees) < 3
        if rotheadPositioned and neckPositioned:
            return True
        if time.time() - startPositioningTime > 2:
            config.log(
            f"head did not move into requested position (yaw={headImu.yaw}, pitch={headImu.pitch})")
            return False


def monitorLoop():

    checkGround = False
    checkAhead = True
    checkWall = False

    config.log(f"obstacle monitoring thread is running")
    d415 = config.cams[mg.D415]
    headImu = config.marvinShares.cartDict.get(mg.SharedDataItem.HEAD_IMU)

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
                config.inForwardMove = False
                continue

        if checkGround:

            config.log(f"D415 ground check")
            config.log(f"ground check with rotheadDegrees: 0, neckDegrees: {mg.pitchGroundWatchDegrees}")

            if not setHeadDegrees(0, mg.pitchGroundWatchDegrees):
                continue


            if not d415.takeDepth():
                config.log(f"could not acquire depth points")
                continue

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


        if checkAhead:   # check medium distance

            config.log(f"check ahead obstacles with rotheadDegrees: 0, neckDegrees {mg.pitchAheadWatchDegrees}")

            if not setHeadDegrees(0, mg.pitchAheadWatchDegrees):
                continue

            freeMoveDistance = 0

            if not d415.takeDepth():
                config.log(f"could not acquire the depth points")
                continue

            else:
                points = processDepth.alignPointsWithGround(d415.depth, mg.pitchAheadWatchDegrees)

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

            config.log(f"depth wall check with rotheadDegrees: 0, neckDegrees: {mg.pitchWallWatchDegrees}")

            if not setHeadDegrees(0, mg.pitchWallWatchDegrees):
                continue

            if not d415.takeDepth():
                config.log(f"could not acquire depth points")
                continue

            distClosestObstacle, obstacleDirection = findObstacles.distGroundObstacles(d415.depth)


            if distClosestObstacle > 0.4:
                config.log(f"continue driving, free move: {distClosestObstacle * 100:.0f} cm")
                checkGround = True      # revert back to ground check
                checkWall = False

            else:
                config.log(
                    f"stop move forward because of a wall obstacle ahead in {distClosestObstacle * 100:.0f} cm at {obstacleDirection:.0f} degrees")
                config.cartCommandMethods.stopCart(config.marvinShares.cartRequestQueue, "wall object detected")
                config.flagInForwardMove = False

