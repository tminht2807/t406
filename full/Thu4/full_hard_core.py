
import cv2
import numpy as np
import threading
import time
import serial
import matplotlib.pyplot as plt
import math
from tensorflow.lite.python.interpreter import Interpreter

from picamera2 import Picamera2


class All_detection(threading.Thread):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.current_command = None  # Biến lưu trữ lệnh hiện tại
        # Khởi tạo đối tượng detector
        self.model_path = '/home/admin/Downloads/obstacle/tf_lite/full/detect.tflite'
        self.label_path = '/home/admin/Downloads/obstacle/tf_lite/full/labelmap.txt'
        self.min_confidence = 0.5
        
     
        self.ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
        
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.current_command = "none"
        self.float_input = (self.input_details[0]['dtype'] == np.float32)
        
        with open(self.label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    stop_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade.xml")
    parking_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/parking_cascade.xml")
    cross_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade_crosswalk.xml")
    straight_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade_straight.xml")


    #trafic light: 
    Red_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/red.xml")
    Yellow_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/yellow.xml")
    Green_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascadegreen.xml")


    def run(self):
        cv2.startWindowThread()

        while True:
            frame = self.cap.capture_array()
            #RoI_detection = frame[0:300, 320:640]


            
            # Process the obstacle detection: 
            frame_output,signal2=self.detect_objects(frame)    
            if signal2=='none':
                self.send_command_signal(1,10)

            # Process the frame (detect signs)
            sign_signal, dist_sign = self.sign_detection(frame)

            if sign_signal == 'Stops':
                if dist_sign < 15:
                    self.send_command_signal(1, 0)
                    time.sleep(5)
                    self.send_command_signal(1, 10)
                    time.sleep(2)
            
            elif sign_signal == 'Parkings':
                if dist_sign < 10 and dist_sign > 0:
                    self.send_command_signal(1, 5)
            
            elif sign_signal == 'None':
                self.send_command_signal(1, 10)  # Example: normal operation
            
            #process the trafficlight detection: 
            light_signal, dist_red_light = self.Traffic_Light_detection(frame)

            if light_signal == 'Red Light':
                if dist_red_light < 10 and dist_red_light > 5:
                    self.send_command_signal(1, 0)
                    time.sleep(5)
                    self.send_command_signal(1, 10)
                    time.sleep(2)
            
            elif light_signal == 'Yellow Light':
                self.send_command_signal(1, 5)
            
            elif light_signal == 'None':
                self.send_command_signal(1, 10)  # Example: normal operation


                
            
            # Display the frame
#             cv2.imshow('Frame_Sign', frame)
            
            # Check for exit key
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        
        # Release the capture
        self.cap.release()
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

        for (x, y, w, h) in parkings:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Parking Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            dist_Sign = int((-0.4412)*w+63.9706)
            cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        
        return signal, dist_Sign
    
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
        signal = 'No Signal'
        
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
    
    def detect_objects(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)
        
        if self.float_input:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        signal='none'
        for i in range(len(scores)):
            if scores[i] > self.min_confidence and scores[i] <= 1.0:
                ymin = int(max(1, boxes[i][0] * imH))
                xmin = int(max(1, boxes[i][1] * imW))
                ymax = int(min(imH, boxes[i][2] * imH))
                xmax = int(min(imW, boxes[i][3] * imW))

                object_name = self.labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                distance = int((-0.4412)*int(xmax - xmin)+63.9706)


                
                if int(classes[i])==0:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, f'Name:{label},Width: {xmax - xmin:.2f}, Height: {ymax - ymin:.2f}, Distance: {distance}', (xmin, label_ymin-7),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    signal=classes[i]
                if int(classes[i])==1:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]

                if signal!='none':    
                    if distance < -50:
                        self.send_command_signal(1,0)
                        time.sleep(2)
        return(frame,signal)


    
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
            


# # Mở video capture
# cap = cv2.VideoCapture(0) #D:\\OKhe\\bosch\\Traffic_sign\\object.mp4
# # Tạo một đối tượng ObjectDetector
# detector = All_detection(cap)
# # Bắt đầu luồng
# detector.start()