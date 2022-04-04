import math

import cv2 as cv
import numpy as np


class simulation:
    def __init__(self):
        self.xSize = 1000
        self.ySize = 1000


        #circle track parameters
        self.lineWidth = 5
        self.lineColor = (255, 255, 255)
        self.edgeBufferSize = 100
        self.trackWidth = 200

        #draw
        self.carColor = (255, 0, 0)
        self.carThickness = 5
        self.carSize = 15

        #race params
        self.speedMovePixels = 20
        self.changeAngle = 5


    def generateCircleCourse(self):
        theCourse = np.zeros((self.ySize, self.xSize))

        cv.circle(theCourse, (int(self.xSize/2), int(self.ySize/2)), min(int(self.xSize/2), int(self.ySize/2)) - self.edgeBufferSize, self.lineColor, thickness=self.lineWidth)

        cv.circle(theCourse, (int(self.xSize / 2), int(self.ySize / 2)),
                  min(int(self.xSize / 2), int(self.ySize / 2)) - self.edgeBufferSize - int(self.trackWidth /2), self.lineColor,
                  thickness=self.lineWidth)

        return theCourse

    def getXYChange(self, angle):

        smallAngle = angle % 180
        if smallAngle > 90:
            # smallAngle = 90 - (smallAngle - 90)
            if smallAngle < 45:
                smallAngle = 90 - (smallAngle - 90)
            else:
                smallAngle = smallAngle

        if angle > 0:
            quadrant = 0
            tmpAngle = abs(smallAngle)
            while tmpAngle < abs(angle):
                tmpAngle += 90
                quadrant += 1

            quadrant = quadrant%4
        else:
            quadrant = 0
            tmpAngle = abs(smallAngle)
            while tmpAngle > abs(angle):
                tmpAngle -= 90
                quadrant -= 1

            quadrant = quadrant % 4
            quadrant = 4 - quadrant

        xChange = int(math.sin(smallAngle * 2 * math.pi / 360) * self.speedMovePixels)
        yChange = int(math.cos(smallAngle * 2 * math.pi / 360) * self.speedMovePixels)

        if quadrant == 0 or quadrant == 1:
            xChange = -xChange

        if quadrant == 0 or quadrant == 3:
            yChange = -yChange

        print("angle", angle, "xchange", xChange, "ychange", yChange, "small angle", smallAngle, "quadrant", quadrant, "math.sin", math.sin(smallAngle),
              "math.cos", math.cos(smallAngle))
        return xChange, yChange

    def centerToPoint(self, point, angle, theCourse, color, draw=True):
        xChange, yChange = self.getXYChange(angle)

        firstPoint = (point[0] - xChange, point[1] - yChange)
        secondPoint = (point[0] + xChange, point[1] + yChange)
        if draw:
            cv.arrowedLine(theCourse, secondPoint, firstPoint, color, self.carThickness)
        else:
            return firstPoint, secondPoint

    def drawPoint(self, nextPoint, angle, prevPoint, prevAngle, theCourse):

        # def drawHelper(point, color):
            # firstPoint = (point[0] - int(self.carSize / 2), point[1] - int(self.carSize / 2))
            # secondPoint = (point[0] + int(self.carSize / 2), point[1] + int(self.carSize / 2))
            # cv.arrowedLine(theCourse, firstPoint, secondPoint, color, self.carThickness)

        #erase prevPoint
        if prevPoint:
            self.centerToPoint(prevPoint, prevAngle, theCourse, (0,0,0))
            # drawHelper(prevPoint, (0, 0, 0))

        self.centerToPoint(nextPoint, angle, theCourse, self.carColor)
        # drawHelper(nextPoint, self.carColor)

        cv.imshow("theCourse", theCourse)
        cv.waitKey(1)

    def getAngle(self, angle):
        userResponse = input("l,r,s?")
        if userResponse == "l":
            angle = angle - self.changeAngle
        elif userResponse == "r":
            angle = angle + self.changeAngle
        else:
            angle = angle

        return angle

    def calculateNextPoint(self, point, angle):
        xChange, yChange = self.getXYChange(angle)

        return (point[0] + xChange, point[1] + yChange)

    def checkBoundary(self, point, course, angle):
        firstPoint, secondPoint = self.centerToPoint(point, angle, None, None, draw=False)
        # firstPoint = (point[0] - int(self.carSize / 2), point[1] - int(self.carSize / 2))
        # secondPoint = (point[0] + int(self.carSize / 2), point[1] + int(self.carSize / 2))

        print(course[firstPoint])
        print(course[secondPoint])

        if course[firstPoint] or course[secondPoint]:
            print("CRASH!!!")
            return True

        return False


    def race(self):
        crash = False
        firstPoint = (750, 750)
        prevPoint = None
        prevAngle = None
        theCourse = self.generateCircleCourse()
        background = theCourse.copy()
        angle = -30.1#0.1

        while not crash:
            self.drawPoint(firstPoint, angle, prevPoint, prevAngle, theCourse)

            angle = self.getAngle(angle)

            prevPoint = firstPoint
            prevAngle = angle
            firstPoint = self.calculateNextPoint(firstPoint, angle)

            # crash = self.checkBoundary(firstPoint, background, angle)



if __name__ == "__main__":
    mySimulation = simulation()
    mySimulation.race()
