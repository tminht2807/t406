import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import serial
import time
import threading
from picamera2 import Picamera2

from class_climbing_and_curving import AutonomousCarController
from full_hard_core import All_detection
from class_obstacle_tflite import ObjectDetector
from class_bien_bao_full import Sign_detection
from class_den_giao_thong_4 import light_detection
from class_lanekeeping_4 import LaneDetectionThread

    
if __name__ == "__main__":
    cv2.startWindowThread()    
    cap1 = Picamera2()
    cap1.configure(cap1.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    cap1.start()

#     cap1 = cv2.VideoCapture(0) "/home/admin/Downloads/Autonomous_vehicle/drive2.mp4"
    # Create instances of MyThread1 and CameraCaptureThread
    lane_thread=LaneDetectionThread(cap1)
    detection_thread = All_detection(cap1) 
    #IMU=AutonomousCarController()
#     thread_obstacle = ObjectDetector(cap1)
#     sign_thread = Sign_detection(cap1)
#     light_thread = light_detection(cap1)


    # Start both threads
    lane_thread.start()
    detection_thread.start()
#     IMU.start()
#     thread_obstacle.start()
#     sign_thread.start()
#     light_thread.start()

      
    # Wait for both threads to finish
    lane_thread.join()
    detection_thread.join()
#     IMU.join()
#     thread_obstacle.join()
#     sign_thread.join()
#     light_thread.join()
    
    print("All threads have finished")   
