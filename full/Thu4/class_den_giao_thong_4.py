import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import serial
import time
import threading

class light_detection(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
        self.current_command = None

    # Khởi tạo CascadeClassifier và đọc tệp xml cho đèn giao thông màu đỏ, vàng và xanh
    Red_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/red.xml")
    Yellow_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/yellow.xml")
    Green_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascadegreen.xml")

    def run(self):
        cv2.startWindowThread()

        while True:
#             frame = self.cap.capture_array()
            _,frame = self.cap.read()
            
            # Process the frame (detect signs)
            sign_signal, dist_red_light = self.Traffic_Light_detection(frame)

            if sign_signal == 'Red Light':
                if dist_red_light < 10 and dist_red_light > 5:
                    self.send_command_signal(1, 0)
            elif sign_signal == 'Yellow Light':
                    self.send_command_signal(1, 5)
            
            elif sign_signal == 'None':
                self.send_command_signal(1, 15)  # Example: normal operation
            
            # Display the frame
#             cv2.imshow('Frame_traffic_light', frame)
            
            # Check for exit key
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        
        # Release the capture
        cv2.destroyAllWindows()


    def Traffic_Light_detection(self,frame):
        # Chuyển đổi frame sang ảnh xám
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Phát hiện đèn giao thông màu đỏ
        red_lights = self.Red_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Phát hiện đèn giao thông màu vàng
        yellow_lights = self.Yellow_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Phát hiện đèn giao thông màu xanh
        green_lights = self.Green_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Mặc định là không có tín hiệu
        signal = 'None'
        
        # Kiểm tra tín hiệu đèn giao thông
        if len(red_lights) > 0:
            signal = 'Red Light'
            print("Traffic Light Signal:", signal)
        elif len(yellow_lights) > 0:
            signal = 'Yellow Light'
            print("Traffic Light Signal:", signal)
        elif len(green_lights) > 0:
            signal = 'Green Light'
            print("Traffic Light Signal:", signal)
        
        # In tín hiệu
        #print("Traffic Light Signal:", signal)
        
        # Vẽ hình chữ nhật và thêm văn bản cho mỗi loại đèn giao thông được phát hiện
        dist_red= 0 
        dist_yellow= 0 
        dist_green=0
        for (x, y, w, h) in red_lights:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Red Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            dist_red = int((-0.4412)*w+63.9706)
            cv2.putText(frame, 'D = {} cm'.format(dist_red), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        for (x, y, w, h) in yellow_lights:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, 'Yellow Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            dist_yellow = int((-0.4412)*w+63.9706)
            cv2.putText(frame, 'D = {} cm'.format(dist_yellow), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

        for (x, y, w, h) in green_lights:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Green Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            dist_green = int((-0.4412)*w+63.9706)
            cv2.putText(frame, 'D = {} cm'.format(dist_green), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

        return [signal, dist_red]
    

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
