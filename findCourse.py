import cv2 as cv
import numpy as np


class findCourse:
    def __init__(self):

        self.myKernel = self.makeCircularKernel(10)


    def fillItUp(self, image):
        h, w = image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        cv.floodFill(image, mask, (0, 0), 255)
        cv.floodFill(image,mask, (int(w/2), int(h/2)), 255)

        return image

    def makeCircularKernel(self, diameter):
        preK = np.zeros((diameter, diameter, 3), np.uint8)
        r = diameter // 2
        preK = cv.circle(preK, (r, r), r, (255, 255, 255), thickness=-1)
        kernel = preK[:, :, 0]
        return kernel

    def findTrack_oneFrame(self, image):
        # img = cv.imread('gradient.png', 0)
        thisOne = image[:,:,1]
        noGreen = image[:,:,2]
        # gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        ret, thresh1 = cv.threshold(noGreen, 125, 255, cv.THRESH_BINARY_INV)
        thresh1 = cv.dilate(thresh1, self.myKernel)
        cv.imshow("thresh1",thresh1)
        fillIt = self.fillItUp(thresh1.copy())

        return fillIt

    def doAll(self):
        image = cv.imread("./data/testPic2.jpg")

        fillIt = self.findTrack_oneFrame(image)

        cv.imshow("orig", image)
        # cv.imshow("test", thresh1)
        # cv.imshow("thisOne", thisOne)
        # cv.imshow("noGreen", noGreen)
        cv.imshow("fillIt", fillIt)

        cv.waitKey(0)

if __name__ == "__main__":
    seeker = findCourse()
    seeker.doAll()



