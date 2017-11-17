import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

with open("train.json","r") as f:
    load_dict = json.load(f)
    #print(load_dict)
    print(type(load_dict[1]))
    print(len(load_dict[1]))
    print(load_dict[1].keys())

img_list1 = []
img_list2 = []

"""for i in load_dict[1]["band_2"]:
    img_list.append(10**(i/10))"""

canvas = np.zeros((75,75,3), dtype="float")

"""========================================================"""

maxnumbers1 = []
minnumbers1 = []

for i in range(len(load_dict)):
    maxnumbers1.append(max(load_dict[i]["band_1"]))
    minnumbers1.append(min(load_dict[i]["band_1"]))

maxnumber1 = max(maxnumbers1)
minnumber1 = min(minnumbers1)

print(maxnumber1)
print(minnumber1)
ladder_length1 = (maxnumber1 - minnumber1)/255

for i in load_dict[0]["band_1"]:
    img_list1.append(int((i-minnumber1)/ladder_length1))

for i in range(75):
    for j in range(75):
        canvas[i][j][0] = img_list1[75*i+j]

canvas1 = np.zeros((75,75,1), dtype="float") 
for i in range(75):
    for j in range(75):
        canvas1[i][j] = img_list1[75*i+j]

"""========================================================="""

maxnumbers2 = []
minnumbers2 = []

for i in range(len(load_dict)):
    maxnumbers2.append(max(load_dict[i]["band_2"]))
    minnumbers2.append(min(load_dict[i]["band_2"]))

maxnumber2 = max(maxnumbers2)
minnumber2 = min(minnumbers2)

print(maxnumber2)
print(minnumber2)
ladder_length2 = (maxnumber2 - minnumber2)/255

for i in load_dict[0]["band_1"]:
    img_list2.append(int((i-minnumber2)/ladder_length2))

for i in range(75):
    for j in range(75):
        canvas[i][j][1] = img_list2[75*i+j]

canvas2 = np.zeros((75,75,1), dtype="float") 
for i in range(75):
    for j in range(75):
        canvas2[i][j] = img_list2[75*i+j]

"""==========================================================="""

#print(canvas)

cv2.imwrite("Canvas1.jpg",canvas1)
cv2.imwrite("Canvas2.jpg",canvas2)
cv2.imwrite("Canvas.jpg",canvas)
#cv2.waitKey(0)

print(load_dict[0]["is_iceberg"])
