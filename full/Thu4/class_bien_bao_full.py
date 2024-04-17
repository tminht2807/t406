import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import serial
import time
import threading


class Sign_detection(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
        self.current_command = None

    stop_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade.xml")
    parking_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/parking_cascade.xml")
    cross_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade_crosswalk.xml")
    straight_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade_straight.xml")

    def run(self):
        while True:
#             frame = self.cap.capture_array()
            _,frame = self.cap.read()
#              RoI_detection=frame[0:300,320:640]
            
            # Process the frame (detect signs)
            sign_signal, dist_sign = self.sign_detection(frame)

            if sign_signal == 'Stops':
                 if dist_sign <30:
                    self.send_command_signal(1, 0)
                    time.sleep(5)
                    self.send_command_signal(1, 15)
                    time.sleep(2)
            
            elif sign_signal == 'cross':
                 if dist_sign < 25 and dist_sign > 0:
                     self.send_command_signal(1,3)   
#                     self.send_command_signal(1,0)
#                     time.sleep(2)
#                     self.send_command_1(10,12,0)
#                     time.sleep(14)
#                     self.send_command_1(-10,5,23)
#                     time.sleep(7)
#                     self.send_command_1(-12,3,-23)
#                     time.sleep(5)
#                     self.send_command_1(12,3,-23)
#                     time.sleep(5)
#                     self.send_command_1(10,3,23)
#                     time.sleep(6)
#                     self.send_command_1(10,10,0)
#                     time.sleep(11)
                    
            elif sign_signal == 'None':
                self.send_command_signal(1, 6)  # Example: normal operation
                time.sleep(0.05)
            # Display the frame
#             cv2.imshow('Frame', frame)
            
            # Check for exit key
            key = cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the capture

        cv2.destroyAllWindows()

    def sign_detection(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stops = self.stop_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        parkings = self.parking_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        cross = self.cross_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        straight = self.straight_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        signal = 'None'
        dist_Sign=0

        if len(stops) > 0:
            signal = 'Stops'
            print("Sign Signal:", signal)
        elif len(parkings) > 0:
            signal = 'Parkings'
            print("Sign Signal:", signal)
        elif len(cross)>0: 
            signal = 'cross'
        elif len(straight)>0: 
            signal = 'straight'

        for (x, y, w, h) in stops:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Stop Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            dist_Sign = int((-0.4412)*w+63.9706)
            cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
            
        for (x, y, w, h) in cross:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, 'Cross Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            dist_Sign = int((-0.4412)*w+63.9706)
            cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
            
        for (x, y, w, h) in straight:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, 'Straight Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            dist_Sign = int((-0.4412)*w+63.9706)
            cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

        for (x, y, w, h) in parkings:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Parking Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            dist_Sign = int((-0.4412)*w+63.9706)
            cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        
        return signal, dist_Sign

    
    def send_command_signal(self,msgID,angle):
        if angle != self.current_command:  # Kiểm tra xem lệnh mới có khác với lệnh hiện tại hay không
            self.current_command = angle  # Cập nhật lệnh hiện tại
            # Send the command with the new angle
            command = f"#{msgID}:{self.current_command};;\r\n"  # Structure of command from RPi
            self.ser.write(command.encode())
            time.sleep(0.5)  # Delay for Nucleo to response
            response = self.ser.readline().decode().strip()
            print(f"Response from Nucleo: {response}")
            # Xóa giá trị đã gửi khỏi hàng đợi# Khởi tạo webcam
    def send_command_1(self, speed, time, angle):
        command = f"#9:{speed};{time};{angle};;\r\n"  
        self.ser.write(command.encode())
         
        response = self.ser.readline().decode().strip()
        print(f"Response from Nucleo: {response}")








