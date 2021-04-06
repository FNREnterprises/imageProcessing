
import time
import numpy as np
import cv2
import imutils
import pyrealsense2 as rs

import config


class Camera(object):

    def __init__(self, properties):
        self.name = properties['name']
        self.deviceId = properties['deviceId']
        self.cols = properties['cols']
        self.rows = properties['rows']
        self.fovH = properties['fovH']
        self.fovV = properties['fovV']
        self.rotate = properties['rotate']
        self.numReads = properties['numReads']
        self.colAngle = self.fovH / self.cols
        self.rowAngle = self.fovV / self.rows
        self.imgXCenter = int(self.cols/2)

        self.handle = None
        self.image = None

    def saveImage(self, path):
        cv2.imwrite(path, self.image)


class UsbCamera(Camera):

    def __init__(self, properties):
        super().__init__(properties)

        try:
            self.handle = cv2.VideoCapture(self.deviceId, cv2.CAP_DSHOW)
        except Exception as e:
            config.log(f"could not connect with cam {self.name}")
            self.handle = None


    def getResolution(self):
        return self.cols, self.rows


    def takeImage(self, showMillis:int=0):

        config.log(f"connect with {self.name}")

        if self.handle is None:
            config.log(f"{self.name}, deviceId: {self.deviceId} not available", publish=False)
            return None

        for i in range(5):
            _, _ = self.handle.read()
            time.sleep(0.1)
        success, self.image = self.handle.read()

        if success:
            config.log(f"{self.name}, deviceId: {self.deviceId} - image successfully taken")
            #self.handle.release()
            if self.rotate != 0:
                self.image = imutils.rotate_bound(self.image, self.rotate)

            if showMillis > 0:
                cv2.imshow(f"usb image {self.name},{self.deviceId}", self.image)
                cv2.waitKey(showMillis)
                cv2.destroyAllWindows()

            return True
        else:
            config.log(f"{self.name}, deviceId: {self.deviceId} - failed to capture image")
            self.handle.release()
            self.handle = None
            return False


    def loadImage(self, file):
        self.image = cv2.imread(file)


class D415Camera(Camera):
    # class for rgb and depth images

    def __init__(self, properties):
        super().__init__(properties)

        self.D415config = None
        self.streaming:bool = False

        self.pc = rs.pointcloud()
        self.colorizer = rs.colorizer()
        self.depth = None
        self.depthColored = None

        self.D415_Z = 0.095  # meters, cam position above neck axis
        self.D415_Y = 0.15  # meters, cam distance in front of neck axis
        self.D415_MountingPitch = -29  # degrees, cam is mounted with a downward pointing angle
        self.distOffsetCamFromCartFront = 0.1  # meters

        # Configure depth and color streams
        if self.handle is None:
            try:
                self.handle = rs.pipeline()
                self.D415config = rs.config()
                self.D415config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                self.D415config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                #config.log(f"{name} ready to stream")

            except Exception as e:
                config.log(f"could not initialize D415 cam, {e}")
                return

        if not self.streaming:
            self.startStream()

        if self.streaming:
            frames = self.handle.wait_for_frames()
            self.stopStream()


    def startStream(self):
        if not self.streaming:
            try:
                self.handle.start(self.D415config)
                self.streaming = True
                config.log(f"start D415 stream successful")
                return True

            except Exception as e:
                config.log(f"not able to start D415 stream: {e}")
                D415Camera.handle = None
                return False


    def stopStream(self):
        if self.streaming:
            self.handle.stop()
            config.log(f"D415 stream stopped")
        self.streaming = False


    def takeImage(self, showMillis=0):

        if not self.streaming:
            if self.startStream():

                for i in range(self.numReads):      # let cam adapt to current light situation
                    frames = self.handle.wait_for_frames()
                    _ = frames.get_color_frame()
                    time.sleep(0.1)
            else:
                config.log(f"D415 startStream failed")
                return False

        frames = self.handle.wait_for_frames()

        self.colorFrame = frames.get_color_frame()

        if self.colorFrame is None:
            config.log(f"could not acquire colorFrame from D415")
            self.image = None
            return False

        else:
            self.image = np.asanyarray(self.colorFrame.get_data())
            config.log(f"D415 rgb available")
            if showMillis > 0:
                cv2.imshow(f"D415 rgb", self.image)
                cv2.waitKey(showMillis)
                cv2.destroyAllWindows()
            return True


    def takeDepth(self, showMillis=0) -> bool:

        if not self.streaming:
            if self.startStream():
                for i in range(self.numReads):  # let cam adapt to current light situation
                    frames = self.handle.wait_for_frames()
                    _ = frames.get_depth_frame()
                    time.sleep(0.1)
            else:
                config.log(f"D415 startStream failed")
                return False

        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 3)

        frames = self.handle.wait_for_frames()

        depthFrame = frames.get_depth_frame()
        if depthFrame is None:
            config.log(f"could not acquire depth from D415")
            return False

        else:
            decimated = decimate.process(depthFrame)    # make 428x240x3 from 1280x720x3

            # Grab new intrinsics (may be changed by decimation)
            # depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            # w, h = depth_intrinsics.width, depth_intrinsics.height
            pointCloud = self.pc.calculate(decimated)

            # Pointcloud data to arrays
            verts = pointCloud.get_vertices()
            self.depth = np.asanyarray(verts).view(np.float32).reshape(-1, 3)  # xzy

            # Colorize depth frame to jet colormap
            depth_color_frame = self.colorizer.colorize(depthFrame)

            # Convert depth_frame to numpy array to render image in opencv
            self.depthColored = np.asanyarray(depth_color_frame.get_data())

            #depthImage = np.asanyarray(depthFrame.get_data())
            #depthColored = cv2.applyColorMap(cv2.convertScaleAbs(depthImage), cv2.COLORMAP_MAGMA) #.COLORMAP_RAINBOW)
            if showMillis > 0:
                cv2.imshow(f"D415 depth", self.depthColored)
                cv2.waitKey(showMillis)
                cv2.destroyAllWindows()

            config.log(f"D415 depth points available")
            return True

    def getDepthXyz(self):
        # the points in the bird view image (rotated points)
        rows = 240
        cols = 428
        return np.reshape(self.depth, (rows, cols, 3))

    def isCamStreaming(self):
        return self.streaming

#    def saveImage(self):
#        cv2.imwrite(f"initialCamImages/{self.name}Rgb.jpg", self.image)


    def saveDepth(self, path):
        cv2.imwrite(f"initialCamImages/{self.name}DepthColored.jpg", self.depthColored)
        cv2.imwrite(f"initialCamImages/{self.name}Depth.jpg", self.depth)


    def loadImage(self, file):
        self.image = cv2.imread(file)
