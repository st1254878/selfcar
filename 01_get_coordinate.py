import cv2
import numpy as np
import json
import argparse

polygons = []

polygons_dict = {}


def draw_circle(event, x, y, flags, param):
    global polygons

    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(origin_img, (x, y), 5, (0, 0, 255), -1)
        print(int(x/0.75), int(y/0.75))
        polygons.append([int(x/0.75), int(y/0.75)])
        
        
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                default="videos/parking_lot_4.mp4",
                help="path to video directory")
args = vars(ap.parse_args())  # 將parse argument變成dictionary裡面的key-value pair


vid = cv2.VideoCapture(args["video"])
return_value, frame = vid.read()
origin_img = frame.copy()

print(int(origin_img.shape[1]*0.75), int(origin_img.shape[0]*0.75))
origin_img = cv2.resize(
    origin_img, (int(origin_img.shape[1]*0.75), int(origin_img.shape[0]*0.75)))

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', origin_img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        print(polygons)
        polygons_dict["Coordinate"] = polygons
        break

with open('coefficient.json', "w") as outfile:
    json.dump(polygons_dict, outfile)

cood = json.load(open('coefficient.json'))
print(cood)
