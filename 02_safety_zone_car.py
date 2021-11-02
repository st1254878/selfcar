import sys
import numpy as np
import cv2
import time
import os
from argparse import ArgumentParser
from openvino.inference_engine import IECore
import matplotlib.pyplot as plt
import json

def detector(cood, pts):
    # print(pts)
    result = cv2.pointPolygonTest(pts, cood, False)
    print(cood)
    if result > -1:
        return("Alarm!!")
    else:
        return("Nothing Happened")
    
def detect_vehicle(frame, execNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (768, 768), (0, 0, 0))

    # 進行vehicle detection
    detections  = execNet.infer({input_blob: blob})

    vehicles = []
    locs = []

    for i in range(0, net.outputs[out_blob].shape[2]):
        # detections的output長相
        # [1,1,N,7]
        # 最後一維度7個數值分別代表：[image_id, label, conf, x_min, y_min, x_max, y_max]
        confidence = detections['detection_out'][0, 0, i, 2]
        label = detections['detection_out'][0, 0, i, 1]
        # 偵測人, label = 1
        if confidence > 0.2 and int(label) == 2:
            box = detections['detection_out'][0, 0, i, 3:7] * np.array([w, h, w, h])  # 還原原圖的x, y坐標位置
            (startX, startY, endX, endY) = box.astype("int")

            # extract the vehicle ROI
            vehicle = frame[startY:endY, startX:endX]
            vehicles.append(vehicle)
            locs.append((startX, startY, endX, endY))

    return vehicles, locs

# 設定inference engine
print("[INFO] create inference engine...")
ie = IECore()

# 讀取模型IR檔
model_xml = "models/intel/FP16/person-vehicle-bike-detection-crossroad-0078/person-vehicle-bike-detection-crossroad-0078.xml"
model_bin = "models/intel/FP16/person-vehicle-bike-detection-crossroad-0078/person-vehicle-bike-detection-crossroad-0078.bin"
net = ie.read_network(model=model_xml, weights=model_bin)

# 檢視一下，是否使用device執行時，有不支援的layers
supported_layers = ie.query_network(network=net, device_name="MYRIAD")
unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]

if len(unsupported_layers) != 0:
    print("Following layers are not supported by the specified plugin {}:\n {}".format("MYRIAD", ", ".join(unsupported_layers)))
    # sys.exit(1)
else:
    print("All layers are supported")
    

# 設定模型Input與Output的端口
print("[INFO] preparing input/output blobs...")
try:
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
except:
    print("Error")
    
print(net.inputs[input_blob].shape)
print(net.outputs[out_blob].shape)

# 設定模型Batch Size並讀取模型要求的圖片長
net.batch_size = 1
(n, c, h, w) = net.inputs[input_blob].shape
net.reshape({input_blob: (n, c, h*0.75, w*0.75)})


# load model to the plugin
print("[INFO] loading model to the plugin...")
execNet = ie.load_network(network=net, device_name = "MYRIAD")


#define the security area
cood = json.load(open('coefficient.json'))
pts = np.array(cood["Coordinate"], np.int32)
pts = pts.reshape(-1, 1, 2)
print(pts)

# 進行推論
print("[INFO] inferencing image...")
video_path = "videos/parking_lot_4.mp4"
vid = cv2.VideoCapture(video_path)

while(vid.isOpened):
    return_value, frame = vid.read()
    start = time.time()
    people, locs = detect_vehicle(frame, execNet)
    # results = execNet.infer({input_blob: blob})
    end = time.time()
    print("[INFO] inference took {:.6f} seconds...".format(end - start))

    for loc in locs:
    # unpack the bounding box and predictions
        (xmin, ymin, xmax, ymax) = loc
        centroid = (int((xmin+xmax)/2), int((ymin+ymax)/2))
        result = detector(centroid, pts)
        if result == "Alarm!!":
            cv2.rectangle(frame, (xmin, ymin),(xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, result, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break