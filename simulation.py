import math
from pathlib import Path

import cv2 as cv
import numpy as np

from carCamera import CarCamera
import random

class simulation:
    def __init__(self):
        self.xSize = 1000
        self.ySize = 1000

        # circle track parameters
        self.lineWidth = 5
        self.lineColor = (255, 255, 255)
        self.edgeBufferSize = 100
        self.trackWidth = 200

        # draw
        self.carColor = (255, 0, 0)
        self.carThickness = 5
        self.carSize = 5

        # race params
        self.speedMovePixels = 20
        self.changeAngle = 5
        self.courses = list(Path("prepared_courses").iterdir())
        self.course_index = 0

    # NOTE!!!! If you wish to create new courses, but still use simulation. Just follow 3 step guide below.
    def generateCircleCourse(self):
        path = self.courses[self.course_index % len(self.courses)]
        image = cv.imread(str(path / "course.jpg"))
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

        return image

    def getXYChange(self, angle):

        # Keeps angle between -360 and 360. Technically don't need. But in case it gets so many laps
        # that it fills up a memory or something.
        if abs(angle) > 360:
            if angle > 0:
                angle = angle - 360
            elif angle < 0:
                angle = angle + 360

        xChange = int(math.sin(angle * 2 * math.pi / 360) * self.speedMovePixels)
        yChange = int(math.cos(angle * 2 * math.pi / 360) * self.speedMovePixels)

        return xChange, yChange

    def centerToPoint(self, point, angle, theCourse, color, draw=True):
        xChange, yChange = self.getXYChange(angle)

        firstPoint = (point[0] - xChange, point[1] - yChange)
        secondPoint = (point[0] + xChange, point[1] + yChange)
        if draw:
            cv.arrowedLine(theCourse, firstPoint, secondPoint, color, self.carThickness)
        else:
            return firstPoint, secondPoint

    def drawPoint(self, nextPoint, angle, prevPoint, prevAngle, theCourse):

        # erase prevPoint
        if prevPoint:
            self.centerToPoint(prevPoint, prevAngle, theCourse, (0, 0, 0))
            # drawHelper(prevPoint, (0, 0, 0))

        self.centerToPoint(nextPoint, angle, theCourse, self.carColor)
        # drawHelper(nextPoint, self.carColor)

        cv.imshow("theCourse", theCourse)
        carView = self.car.getCarView(nextPoint, angle)
        cv.imshow("car view", carView)
        cv.waitKey(1)

    def getAngle(self, angle, programChoice=None):
        if programChoice == None:
            userResponse = input("l,r,s?")
            if userResponse == "l":
                angle = angle - self.changeAngle
            elif userResponse == "r":
                angle = angle + self.changeAngle
            else:
                angle = angle
        else:
            angle = angle + programChoice

        return angle

    def calculateNextPoint(self, point, angle):
        xChange, yChange = self.getXYChange(angle)

        return (point[0] + xChange, point[1] + yChange)

    def checkBoundary(self, point, course, angle):
        firstPoint, secondPoint = self.centerToPoint(point, angle, None, None, draw=False)
        # firstPoint = (point[0] - int(self.carSize / 2), point[1] - int(self.carSize / 2))
        # secondPoint = (point[0] + int(self.carSize / 2), point[1] + int(self.carSize / 2))
        check1 = firstPoint[1],firstPoint[0]
        check2 = secondPoint[1], secondPoint[0]
        try:

            if course[check1] or course[check2]:
                return True
        except IndexError:
            return True
        return False

    def race(self):
        while True:
            crash = False
            firstPoint = self.getStartLocation()
            prevPoint = None
            prevAngle = None
            theCourse = self.generateCircleCourse()
            angle = self.getStartAngle()
            self.background = self.generateCircleCourse()

            self.car = CarCamera(self.background)

            while not crash:
                self.drawPoint(firstPoint, angle, prevPoint, prevAngle, theCourse)

                prevPoint = firstPoint
                prevAngle = angle
                # get next points
                angle = self.getAngle(angle)
                firstPoint = self.calculateNextPoint(firstPoint, angle)
                crash = self.checkBoundary(firstPoint, self.background, angle)
            self.course_index +=1

    def getCarImage(self):
        nextImage = self.car.getCarView(self.firstPoint, self.angle)

        #down size to 15 x 15
        nextImage = cv.resize(nextImage, (15, 15), interpolation=cv.INTER_AREA)

        # insert changing dimensions to be the right size for the network
        nextImage = nextImage.flatten()

        return nextImage

    def reset(self):
        self.crash = False
        self.course_index += 1
        self.firstPoint = self.getStartLocation()
        self.prevPoint = None
        self.prevAngle = None
        self.theCourse = self.generateCircleCourse()
        self.angle = self.getStartAngle()
        self.background = self.generateCircleCourse()
        self.car = CarCamera(self.background)
        firstImage = self.getCarImage()
        return firstImage

    def step(self, action):
        self.drawPoint(self.firstPoint, self.angle, self.prevPoint, self.prevAngle, self.theCourse)

        self.prevPoint = self.firstPoint
        self.prevAngle = self.angle
        # get next points
        networkChoiceAngle = action - 30
        self.angle = self.getAngle(self.angle, networkChoiceAngle)
        self.firstPoint = self.calculateNextPoint(self.firstPoint, self.angle)
        self.crash = self.checkBoundary(self.firstPoint, self.background, self.angle)
        nextImage = self.getCarImage()

        if not self.crash:
            if abs(networkChoiceAngle) < 5:
                #reward for turning less than 5
                reward = 3
            elif abs(networkChoiceAngle) < 10:
                #reward for only turning 10
                reward = 2
            else:
                #reward for not crashing
                reward = 1
        else:
            reward = -3

        return nextImage, reward, (self.crash), None


    def getStartLocation(self):
        path = self.courses[self.course_index % len(self.courses)]
        position = (path / "position.txt").read_text().split(" ")
        return (int(position[0]), int(position[1]))


    def getStartAngle(self):
        path = self.courses[self.course_index % len(self.courses)]
        angles = (path / "angle.txt").read_text().split(" ")
        return int(random.choice(angles))


if __name__ == "__main__":
    mySimulation = simulation()
    mySimulation.race()
