# import cv2
# from detect_objects import detect_objects
# import numpy as np

# def detect_helmets(frame, helmet_model, trapezium):
#     """
#     Detects helmets within the given trapezium bounding box.
    
#     :param frame: The current frame from the video.
#     :param helmet_model: The YOLO model for helmet detection.
#     :param trapezium: List of four (x, y) points defining the trapezium.
#     :return: List of detected helmet and no-helmet bounding boxes inside the trapezium.
#     """
#     x_min = min(p[0] for p in trapezium)
#     y_min = min(p[1] for p in trapezium)
#     x_max = max(p[0] for p in trapezium)
#     y_max = max(p[1] for p in trapezium)
    
#     # Define center and size for getRectSubPix
#     center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
#     size = (x_max - x_min, y_max - y_min)
    
#     # Crop the trapezium region using getRectSubPix
#     roi = cv2.getRectSubPix(frame, (int(size[0]), int(size[1])), center)
#     names = ['Helmet', 'No_Helmet']
    
#     if roi is None or roi.size == 0:
#         return []
    
#     detections = detect_objects(helmet_model, roi)
#     detected_helmets = []
    
#     for detection in detections:
#         print(detection)  # Debugging: Check raw detection values

#         class_name = names[detection['class']]
        
#         # Convert relative coordinates in `roi` to original frame coordinates
#         x1 = int(x_min + (detection['x'] - detection['w'] / 2))  # Convert center-x to x1
#         y1 = int(y_min + (detection['y'] - detection['h'] / 2))  # Convert center-y to y1
#         x2 = int(x_min + (detection['x'] + detection['w'] / 2))  # Convert center-x to x2
#         y2 = int(y_min + (detection['y'] + detection['h'] / 2))  # Convert center-y to y2

#         detected_helmets.append({
#             'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
#             # 'class': class_name
#             'class_name': class_name
#         })
    
#     return detected_helmets



import cv2
from detect_objects import detect_objects
import numpy as np

def detect_helmets(frame, helmet_model, trapezium):
    """
    Detects helmets within the given trapezium bounding box.
    
    :param frame: The current frame from the video.
    :param helmet_model: The YOLO model for helmet detection.
    :param trapezium: List of four (x, y) points defining the trapezium.
    :return: List of detected helmet and no-helmet bounding boxes inside the trapezium.
    """
    x_min = min(p[0] for p in trapezium)
    y_min = min(p[1] for p in trapezium)
    x_max = max(p[0] for p in trapezium)
    y_max = max(p[1] for p in trapezium)
    
    # Define center and size for getRectSubPix
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    size = (x_max - x_min, y_max - y_min)
    
    # Crop the trapezium region using getRectSubPix
    roi = cv2.getRectSubPix(frame, (int(size[0]), int(size[1])), center)
    names = ['Helmet', 'No_Helmet']
    
    if roi is None or roi.size == 0:
        return []
    
    detections = detect_objects(helmet_model, roi)
    return detections

