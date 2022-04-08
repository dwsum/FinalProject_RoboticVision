import cv2 as cv

class findCourse:
    def __init__(self):

        self.image = cv.imread("./data/testPic2.jpg")



    def findTrack_oneFrame(self):
        # img = cv.imread('gradient.png', 0)
        thisOne = self.image[:,:,1]
        noGreen = self.image[:,:,2]
        # gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        noGreen = cv.dilate(noGreen, (3,3))
        ret, thresh1 = cv.threshold(noGreen, 125, 255, cv.THRESH_BINARY_INV)

        cv.imshow("orig", self.image)
        cv.imshow("test", thresh1)
        cv.imshow("thisOne", thisOne)
        cv.imshow("noGreen", noGreen)

        cv.waitKey(0)

if __name__ == "__main__":
    seeker = findCourse()
    seeker.findTrack_oneFrame()



