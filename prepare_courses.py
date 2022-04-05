from pathlib import Path
import numpy as np
import cv2.cv2 as cv

class rawCourse:
    def __init__(self,path):
        self.path = Path("prepared_courses") / path.stem
        self.path.mkdir(exist_ok=True,parents=True)
        self.courseColor = cv.imread(str(path))
        self.course = cv.cvtColor(self.courseColor,cv.COLOR_BGR2GRAY)
        self.startPosition = []
        self.startAngles = []
        self.findStartingPositions()
        self.saveData()

    def findStartingPositions(self):
        min_row_val = 5000000000
        min_row_idx = -1
        for i,row in enumerate(self.course):
            if sum(row)<min_row_val:
                min_row_val = sum(row)
                min_row_idx = i
        columns = np.where(self.course[min_row_idx]==0)
        column_index = columns[0][int(len(columns[0])/2)]
        rows = np.where(self.course[min_row_idx-100:min_row_idx+100,column_index] ==0)
        min_row_idx = rows[0][int(len(rows[0]) / 2)] + min_row_idx-100
        self.startPosition = (column_index,min_row_idx)
        self.startAngles.append(90)
        self.startAngles.append(270)

    def saveData(self):
        cv.imwrite(str(self.path/"course.jpg"),self.course)
        position_path = self.path / "position.txt"
        angle_path = self.path / "angle.txt"
        position_path.write_text("{} {}".format(*self.startPosition))
        angle_path.write_text("{} {}".format(*self.startAngles))



if __name__ == "__main__":
    path = Path("raw_courses")
    for i in path.iterdir():
        rawCourse(i)