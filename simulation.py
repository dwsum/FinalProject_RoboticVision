import math

import cv2 as cv
import numpy as np

from carCamera import CarCamera


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
        self.background = self.generateCircleCourse(mask=True)
        self.car = CarCamera(self.background)


    #NOTE!!!! If you wish to create new courses, but still use simulation. Just follow 3 step guide below.
    def generateCircleCourse(self, mask = False):
        theCourse = np.zeros((self.ySize, self.xSize))

        if not mask:
            theThickness = self.lineWidth
        else:
            #for creating courses, make sure this is the variable used for fill. This should fill whatever shape you do. (step 1)
            theThickness = -1

        #for creating new courses, simply make the outside curve here. (step 2)
        cv.circle(theCourse, (int(self.xSize/2), int(self.ySize/2)), min(int(self.xSize/2), int(self.ySize/2)) - self.edgeBufferSize, self.lineColor, thickness=theThickness)

        if mask:
            theCourse = 255 - theCourse

        #for creating new courses, simply draw the inside curve here (step 3)
        cv.circle(theCourse, (int(self.xSize / 2), int(self.ySize / 2)),
                  min(int(self.xSize / 2), int(self.ySize / 2)) - self.edgeBufferSize - int(self.trackWidth /2), self.lineColor,
                  thickness=theThickness)

        if mask:
            cv.imshow("mask", theCourse)
            cv.waitKey(1)

        return theCourse

    def getXYChange(self, angle):

        # Keeps angle between -360 and 360. Technically don't need. But in case it gets so many laps
        # that it fills up a memory or something.
        if abs(angle) >360:
            if angle > 0:
                angle = angle - 360
            elif angle < 0:
                angle = angle + 360

        xChange = int(math.sin(angle * 2 * math.pi / 360) * self.speedMovePixels)
        yChange = int(math.cos(angle * 2 * math.pi / 360) * self.speedMovePixels)

        print("angle", angle, "xchange", xChange, "ychange", yChange)
        return xChange, yChange


    def centerToPoint(self, point, angle, theCourse, color, draw=True):
        xChange, yChange = self.getXYChange(angle)

        # cv.imshow("colorCourse",theCourse)
        # cv.waitKey()
        firstPoint = (point[0] - xChange, point[1] - yChange)
        secondPoint = (point[0] + xChange, point[1] + yChange)
        if draw:
            cv.arrowedLine(theCourse, firstPoint, secondPoint, color, self.carThickness)
        else:
            return firstPoint, secondPoint

    def drawPoint(self, nextPoint, angle, prevPoint, prevAngle, theCourse):

        #erase prevPoint
        if prevPoint:
            self.centerToPoint(prevPoint, prevAngle, theCourse, (0,0,0))
            # drawHelper(prevPoint, (0, 0, 0))

        self.centerToPoint(nextPoint, angle, theCourse, self.carColor)
        # drawHelper(nextPoint, self.carColor)

        cv.imshow("theCourse", theCourse)
        carView =self.car.getCarView(nextPoint,angle)
        cv.imshow("car view",carView)
        cv.waitKey(1)

    def getAngle(self, angle, programChoice=None):
        if programChoice == None:
            userResponse = input("l,r,s?")
        else:
            if programChoice == 0:
                userResponse = "l"
            else:
                userResponse = "r"

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
        theCourse = self.generateCircleCourse(mask=False)
        angle = -30

        while not crash:
            self.drawPoint(firstPoint, angle, prevPoint, prevAngle, theCourse)

            prevPoint = firstPoint
            prevAngle = angle
            #get next points
            angle = self.getAngle(angle)
            firstPoint = self.calculateNextPoint(firstPoint, angle)
            crash = self.checkBoundary(firstPoint, self.background, angle)


    def reset(self):
        self.crash = False
        self.firstPoint = (750, 750)
        self.prevPoint = None
        self.prevAngle = None
        self.theCourse = self.generateCircleCourse(mask=False)
        self.angle = -30
        firstImage = self.car.getCarView(self.firstPoint, self.angle)

        return firstImage

    def step(self, action):
        self.drawPoint(self.firstPoint, self.angle, self.prevPoint, self.prevAngle, self.theCourse)

        self.prevPoint = self.firstPoint
        self.prevAngle = self.angle
        # get next points
        self.angle = self.getAngle(self.angle, action)
        self.firstPoint = self.calculateNextPoint(self.firstPoint, self.angle)
        self.crash = self.checkBoundary(self.firstPoint, self.background, self.angle)

        #TODO make correct return type


if __name__ == "__main__":
    mySimulation = simulation()
    mySimulation.race()
