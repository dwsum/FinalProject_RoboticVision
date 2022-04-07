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

        return ourList

    def step(self, action):
        action = action - 30
        jsonResult = {"angle": str(action)}
        jsonResult = str.encode(json.dumps(jsonResult))
        # print("jsonResults", jsonResult, type(jsonResult))
        self.sock.send(jsonResult)

        returned = self.sock.recv(16384)
        returned = returned.decode()
        theJson = json.loads(returned)
        ourList = theJson['map0']
        goingForward = theJson['going_forward']

        ourList = np.array(ourList, dtype=np.float)

        while self.goingBack:
            action = action - 30
            jsonResult = {"angle": str(action)}
            jsonResult = str.encode(json.dumps(jsonResult))
            self.sock.send(jsonResult)

            returned = self.sock.recv(16384)
            returned = returned.decode()
            theJson = json.loads(returned)
            ourList = theJson['map0']
            goingForward = theJson['going_forward']

            ourList = np.array(ourList, dtype=np.float)

            if goingForward:
                self.goingBack = False
                break


        if goingForward:
            # if abs(action) < 5:
            #     #reward for turning less than 5
            #     reward = 3
            # elif abs(action) < 10:
            #     #reward for only turning 10
            #     reward = 2
            # else:
            #     #reward for not crashing
            #     reward = 1
            self.reward += 1
            reward = self.reward
        else:
            reward = -3

        ourList = cv.resize(ourList, (15, 15), interpolation=cv.INTER_AREA)

        # insert changing dimensions to be the right size for the network
        ourList = ourList.flatten()

        return ourList, reward, not goingForward, None



