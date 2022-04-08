import json
import socket
import cv2 as cv

import numpy as np


class carInstance:
    def __init__(self):
        try:
            self.sock = socket.socket()
        except socket.error as err:
            print('Socket error because of %s' % (err))

        port = 8080
        address = "10.37.0.8"

        self.prevSteerCol = None
        self.prevList = None

        try:
            print("connecting")
            self.sock.connect((address, port))
            print("connected!")
        except socket.gaierror:

            print('There an error resolving the host')



    def reset(self):
        self.goingBack = True
        self.reward = 1

        ourList, reward, goingForward, _ = self.step(0)

        self.prevSteerCol = None
        self.prevList = None

        return ourList

    def handPickValue(self, downsamplemasked_obstacles):
        numCols = 15
        numRows = 15

        colWeights = np.full((numCols), 1.0)
        if self.prevSteerCol is not None:
            colWeights[self.prevSteerCol - 1] = .8
        rowWeights = np.linspace(0.0, 2.0, numRows) ** 3
        weightMatrix = np.outer(rowWeights, colWeights)

        steeringMatrix = ((downsamplemasked_obstacles.astype(float) * 1) * weightMatrix).astype(int)
        N = int(numCols / 4)
        weightedCols = np.sum(steeringMatrix, axis=0)
        moving_average = np.convolve(weightedCols, np.ones(N) / N, mode='valid')

        lowest_added = np.partition(moving_average, 1)[1]  # np.argmin(moving_average) +N
        lowest = np.where(moving_average == lowest_added)[0][0] + N

        lowestRow = steeringMatrix[:, lowest - 1]
        strength = 0
        for i in reversed(range(numCols)):
            if lowestRow[i]:
                strength = i
                break
        steerDirection = ((lowest / (numCols - 1)) - 0.5) * 2.0 * 30#steerAngle
        speed = (strength / numCols) * 4.0
        steerDirection *= (strength / numCols) * 1.25
        self.prevSteerCol = lowest

        return steerDirection

    def sendResults(self, action, handPicked):
        # jsonResult = {"angle": str(action)}
        jsonResult = {"angle": str(int(handPicked))}
        jsonResult = str.encode(json.dumps(jsonResult))
        # print("jsonResults", jsonResult, type(jsonResult))
        self.sock.send(jsonResult)

        returned = self.sock.recv(16384)
        returned = returned.decode()
        theJson = json.loads(returned)
        ourList = theJson['map0']
        goingForward = theJson['going_forward']

        ourList = np.array(ourList, dtype=np.float)

        return ourList, goingForward

    def step(self, action):
        if self.prevList is not None:
            handPicked = self.handPickValue(self.prevList)
        else:
            handPicked = 0
        action = action - 30
        ourList, goingForward = self.sendResults(action, handPicked)


        while self.goingBack:
            ourList, goingForward = self.sendResults(action, handPicked)

            if goingForward:
                self.goingBack = False
                break


        if goingForward:
            diff = abs(action - handPicked)
            if diff < 3:
                reward = 1
            # elif diff < 10:
            #     reward = 2
            else:
                reward = -1
            # if abs(action) < 5:
            #     #reward for turning less than 5
            #     reward = 3
            # elif abs(action) < 10:
            #     #reward for only turning 10
            #     reward = 2
            # else:
            #     #reward for not crashing
            #     reward = 1
            # self.reward += 1
            # reward = self.reward


        else:
            reward = 0

        ourList = cv.resize(ourList, (15, 15), interpolation=cv.INTER_AREA)
        self.prevList = ourList
        # insert changing dimensions to be the right size for the network
        ourList = ourList.flatten()

        return ourList, reward, not goingForward, None



