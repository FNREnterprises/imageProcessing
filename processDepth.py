
import numpy as np

import config
from marvinglobal import marvinglobal as mg

def savePointCloud(pc, filename):
    # save as pickle file for testing
    myFile = open(filename, 'wb')
    np.save(myFile, pc)
    myFile.close()


def loadPointCloud(filename):
    myFile = open(filename, 'rb')
    pc = np.load(myFile)
    myFile.close()
    return pc


def rotatePointCloud(verts, angle):

    # rotate over x-axis
    cosa = np.cos(np.radians(angle))
    sina = np.sin(np.radians(angle))

    # return rotated point cloud array
    return np.dot(verts, np.array([
        [1., 0, 0],
        [0, cosa, -sina],
        [0, sina, cosa]]))


def calculateCamXYZ(useHeadImu=True):
    '''
    The camera is mounted on the head with a downward pointing angle
    Y and Z of the cam depend on the neck setting (headImuPitch, headImuYaw)
    headYaw is assumed to be 0!
    :return:
    '''
    d415 = config.cams[mg.D415]

    if not useHeadImu:
        # imageAngle: -80.38, camYXZ: (0, 0.26752425602438334, 1.5068158952609565)
        #return -68, (0, 0.267, 1.5)  # ground watch values
        return -45, (0,0.25,1.56) # wall watch values with headNeck -15

    headImu = config.marvinShares.cartDict.get(mg.SharedDataItems.HEAD_IMU)
    neckToCamDist = np.hypot(d415.D415_Y, d415.D415_Z)
    camPosAngle = np.degrees(np.arctan(d415.D415_Z / d415.D415_Y)) + headImu.pitch

    camY = config.robotNeckY + np.cos(np.radians(camPosAngle)) * neckToCamDist
    camZ = config.robotBaseZ + config.robotNeckZ + np.sin(np.radians(camPosAngle)) * neckToCamDist

    camAngle = headImu.pitch + d415.D415_MountingPitch

    config.log(
        f"headcamOrientation: head pitch: {headImu.pitch}, D415 angle: {camAngle:.0f}, camZ: {camZ:.3f}")

    return camAngle, (0, camY, camZ)


def alignPointsWithGround(rawPoints, headPitch, useHeadImu = True):
    '''
    raw points are taken in an angle
    rotate the point cloud to be aligned with the ground
    remove the ground and convert point height to above ground height
    raise below ground points to be an obstacle
    :return: aligned point cloud
    '''
    config.log(f"start align points with ground, headPitch: {headPitch:.0f}")

    # replace out of scope distance values with NaN
    with np.errstate(invalid='ignore'):
        rawPoints[:, 2][(rawPoints[:,2] < 0.2) |
                     (rawPoints[:,2] > 4)] = np.NaN

    #findObstacles.showSliceRaw(rawPoints, 208)

    # get image angle and cam xyz position
    imageAngle, camXYZ = calculateCamXYZ(useHeadImu)

    #showPoints(points,     # rotate points for a horizontal representation of the points
    rotatedPoints = rotatePointCloud(rawPoints, -90-imageAngle)
    config.log(f"rotation angle: {-90-imageAngle:.1f}")

    ####################################################
    # after rotation:
    # points[0] = left/right, center=0
    # points[1] = horizontal distance to point, row 0 farthest
    # points[2] = height of point above ground, in relation to cam height
    ####################################################

    #showPoints(rotatedPoints, 'wall image rotated')

    # subtract the cam height from the points
    rotatedPoints[:,2] = rotatedPoints[:,2] - camXYZ[2]

    # set distance positive
    rotatedPoints[:,1] = -rotatedPoints[:,1]

    # show a rotated slice as line
    #findObstacles.showSlice(rotatedPoints, 208)

    # create obstacles for below ground points, suppress runtime warnings caused by nan values
    with np.errstate(invalid='ignore'):
        rotatedPoints[:,2] = np.where(rotatedPoints[:,2] < - 0.1, 1, rotatedPoints[:,2])

    return rotatedPoints        # for each col/row point the xyz values


