import math

import numpy as np
import cv2 as cv

class CarCamera:
    def __init__(self,background):
        self.IMAGE_H =  480
        self.IMAGE_W = 640
        self.background = background
        src = np.float32([[0, self.IMAGE_H], [self.IMAGE_W, self.IMAGE_H], [0, 0], [self.IMAGE_W, 0]])
        dst = np.float32([[284, self.IMAGE_H], [355, self.IMAGE_H], [0, 0], [self.IMAGE_W, 0]])
        self.Minv = cv.getPerspectiveTransform(dst, src)  # Inverse transformation
        self.bottom_x_offset = 35
        self.top_x_offset = 320
        self.top_y_offset = 480
    def update_background(self,background):
        self.background = background
    def rotatePoint(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point
        angle = -angle +180
        qx = ox + math.cos(angle*np.pi/180) * (px - ox) - math.sin(angle*np.pi/180) * (py - oy)
        qy = oy + math.sin(angle*np.pi/180) * (px - ox) + math.cos(angle*np.pi/180) * (py - oy)
        return qx, qy
    def getCarView(self,point,angle):
        carViewMask = self.getCarMask(point,angle)

        carViewUnwarped = self.rotateAndCropMask(carViewMask,point,angle)

        carView = cv.warpPerspective(carViewUnwarped, self.Minv, (self.IMAGE_W, self.IMAGE_H))  # Image warping
        carViewFilledUp = self.fill(carView)
        return carViewFilledUp

    def getCarMask(self,point,angle):
        pts = np.array(
            [[point[0] - self.top_x_offset, point[1] - self.top_y_offset],
             [point[0] + self.top_x_offset, point[1] - self.top_y_offset],
             [point[0] + self.bottom_x_offset, point[1]], [point[0] - self.bottom_x_offset, point[1]]], np.int32)
        points = []
        for refpoint in pts:
            points.append(self.rotatePoint(point, refpoint, angle))
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        carMask = np.zeros_like(self.background)
        # colorCourse = np.zeros((theCourse.shape[0],theCourse.shape[1],3),dtype=np.int32)
        carMask = cv.polylines(carMask, [pts], True, 255)
        carMask = cv.fillPoly(carMask, [pts], 255)
        return carMask

    def rotateAndCropMask(self, carViewMask, point, angle):
        carView = cv.bitwise_and(carViewMask, self.background)
        image_center = point
        image_center = np.array(image_center,dtype=np.float32)
        rot_mat = cv.getRotationMatrix2D(image_center, -angle + 180, 1.0)
        carViewRotated = cv.warpAffine(carView, rot_mat, carView.shape[1::-1], flags=cv.INTER_LINEAR)
        yminAct = point[1] - self.top_y_offset
        ymin = max(yminAct, 0)
        ymax = point[1]
        xminAct = point[0] - self.top_x_offset
        xmin = max(xminAct, 0)
        xmax = point[0] + self.top_x_offset
        carViewRotatedCropped =  carViewRotated[ymin:ymax, xmin:xmax]
        carView = np.zeros((self.IMAGE_H, self.IMAGE_W))
        if 0 in list(carViewRotatedCropped.shape):
            return carView
        x_start = 0
        y_start = 0
        x_end = carViewRotatedCropped.shape[1]
        y_end = carViewRotatedCropped.shape[0]
        yminAct = point[1] - self.top_y_offset
        xminAct = point[0] - self.top_x_offset
        if yminAct < 0:
            y_start = abs(yminAct)
            y_end += abs(yminAct)
        if xminAct < 0:
            x_start = abs(xminAct)
            x_end += abs(xminAct)
        try:
            carView[y_start:y_end, x_start:x_end] = carViewRotatedCropped
        except:
            pass
        return carView

    def fill(self, carView):
        for i in reversed(range(self.IMAGE_H - 1)):
            carView[i] = np.max(carView[i:i + 2], axis=0)
        return carView


if __name__ == "__main__":
    car = CarCamera()
    car.get_window(12,12,360)