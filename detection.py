
from imageai.Detection import VideoObjectDetection
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

def intersection_over_union(current_box, current_box_name, previous_box, previous_box_name):
    if current_box_name != previous_box_name:
        return 0

    x_max = max(current_box[0], previous_box[0])
    y_max = max(current_box[1], previous_box[1])
    x_min = min(current_box[2], previous_box[2])
    y_min = min(current_box[3], previous_box[3])

    intersection_area = max(0, x_min - x_max) * max(0, y_min - y_max)
    current_box_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
    previous_box_area = (previous_box[2] - previous_box[0]) * (previous_box[3] - previous_box[1])
    result_iou = intersection_area / (current_box_area + previous_box_area - intersection_area)
    return result_iou

def forFrame(frame_number, output_array, output_count, returned_frame):
    print("Previous Frame {}".format(CameraDetector.detections))
    temp_detections = []
    for detection in output_array:

        current_box_points = detection.get('box_points')
        current_box_name = detection.get('name')
        x1 = current_box_points[0]
        y1 = current_box_points[1]
        iou_ids = []
        iou_results = []
        
        for previous_box in CameraDetector.detections:
            previous_box_points = previous_box.get('box_points')
            previous_box_id = previous_box.get('id')
            previous_box_name = previous_box.get('name')
            iou_result = intersection_over_union(current_box_points, current_box_name, previous_box_points, previous_box_name)
            if  iou_result > 0.3:
                iou_results.append(iou_result)
                iou_ids.append(previous_box_id)

        if iou_results:
            id = iou_ids[np.argmax(iou_results)]
        else:
            CameraDetector.id += 1
            id = CameraDetector.id
        test = {'id': id, 'box_points': current_box_points, 'name': current_box_name}
        cv2.putText(returned_frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

        temp_detections.append(test)

    CameraDetector.detections = temp_detections
    print("Current Frame {}".format(CameraDetector.detections))
    cv2.imshow("Detection", returned_frame)
    cv2.waitKey(1)



class CameraDetector:

    detections = []
    id = 0
    def __init__(self, cam):

        self.execution_path = os.getcwd()

        self.color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}
        
        self.resized = False

        self.cap = cv2.VideoCapture(cam)

    def start(self):

        detector = VideoObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(self.execution_path , 
                                            "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detector.detectObjectsFromVideo(camera_input=self.cap, 
                                        output_file_path=os.path.join(self.execution_path, 
                                                                     "video_frame_analysis") , 
                                        frames_per_second=30, 
                                        per_frame_function=forFrame,  
                                        minimum_percentage_probability=70, 
                                        return_detected_frame=True)




if __name__=="__main__":
    cameraDetector = CameraDetector(0)
    cameraDetector.start()



