import numpy as np
import cv2
import time
import threading
import serial
import matplotlib.pyplot as plt
import math

from tensorflow.lite.python.interpreter import Interpreter
from picamera2 import Picamera2
ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1) 



class ObjectDetector(threading.Thread):
    def __init__(self, cap):
        super().__init__()  # Gọi hàm khởi tạo của lớp cha
        # Khởi tạo đối tượng detector
        self.model_path = '/home/admin/Downloads/obstacle/tf_lite/full/detect.tflite'
        self.label_path = '/home/admin/Downloads/obstacle/tf_lite/full/labelmap.txt'
        self.min_confidence = 0.5
        
        self.cap = cap
        self.ser=ser
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
        
    def run(self):
        cv2.startWindowThread()
        while True:
#             frame = self.cap.capture_array()
            _,frame = self.cap.read()
            frame_output,signal2=self.detect_objects(frame)    
            if signal2=='none':
                self.send_command_signal(1,15)

#             cv2.imshow('output',frame_output)        
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break        
        cv2.destroyAllWindows()
        
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


'''
# Khởi tạo và bắt đầu thread
cv2.startWindowThread()
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
detector = ObjectDetector(picam2)
# Bắt đầu luồng
detector.start()
'''


