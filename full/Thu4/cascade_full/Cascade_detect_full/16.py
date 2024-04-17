
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import serial
import time
import threading
from tensorflow.lite.python.interpreter import Interpreter
from picamera2 import Picamera2
ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
steering_angles = deque(maxlen=5)




# Khá»Ÿi táº¡o Ä‘á»‘i tÆ°á»£ng detector
model_path = '/home/admin/Downloads/obstacle/tf_lite/full/detect.tflite'
label_path = '/home/admin/Downloads/obstacle/tf_lite/full/labelmap.txt'
min_confidence = 0.5
        
     
ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
        
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height_image = input_details[0]['shape'][1]
width_image = input_details[0]['shape'][2]
current_command = "none"
float_input = (input_details[0]['dtype'] == np.float32)
        
with open(label_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

stop_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade.xml")
parking_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/parking_cascade.xml")
cross_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade_crosswalk.xml")
straight_cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascade_straight.xml")


#trafic light: 
Red_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/red.xml")
Yellow_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/yellow.xml")
Green_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/obstacle/tf_lite/full/Thu4/cascade_full/Cascade_detect_full/cascadegreen.xml")



def load_image(image_path):
    # Load an image from the specified file
    return cv2.imread(image_path)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # Ä áº£m báº£o line lÃ  máº£ng 1 chiá» u cÃ³ 4 pháº§n tá»­
            if not isinstance(line, np.ndarray):
                line = np.array(line)
            if line.ndim == 2 and line.shape[0] == 1:
                line = line.flatten()
            
            # Kiá»ƒm tra xem line cÃ³ Ä‘á»§ 4 pháº§n tá»­ khÃ´ng
            if len(line) == 4:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            else:
                print("Ä oáº¡n tháº³ng khÃ´ng há»£p lá»‡:", line)

    return line_image


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * 1/2 # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * 1/2 # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))

            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        lane_lines.append(make_points(frame, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):       
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]
   
def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    # reduce the noise of the image: if itâ€™s not between the 5-by-5 Kernel it is deleted
    edges = cv2.Canny(blur, 80, 170) #100 and 200 are min and max gradient intensities
    return edges

def region_of_interest(canny):
    height = canny.shape[0] #first dimension value
    width = canny.shape[1] #second dimension value
    mask = np.zeros_like(canny) #mask initialization
    shape = np.array([[(0, height), (width, height), (width, 300), (0,300)]], np.int32) #polygon 30
    cv2.fillPoly(mask, shape, 255) #mask with polygon size
    masked_image = cv2.bitwise_and(canny, mask) #final result
    cv2.imshow("Canny mainlane", masked_image)
    cv2.waitKey(1)
    return masked_image

def detect_line_segments(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 20
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, np.array([]), minLineLength=60, maxLineGap=200)
    return line_segments

def get_steering_angle(frame, lane_lines):
    height, width, _ = frame.shape
    if len(lane_lines) == 2:
        left_x1, left_y1, left_x2, left_y2 = lane_lines[0][0]
        right_x1, right_y1, right_x2, right_y2 = lane_lines[1][0]
        slope_l=math.atan((left_x1-left_x2) / (left_y1-left_y2))
        slope_r=math.atan((right_x1-right_x2) / (right_y1-right_y2))
        slope_ldeg = int(slope_l * 180.0 / math.pi)
        steering_angle_left = slope_ldeg  
        slope_rdeg = int(slope_r * 180.0 / math.pi)
        steering_angle_right = slope_rdeg
        if left_x2>right_x2: #horizontal line 
            if abs(steering_angle_left) <= abs(steering_angle_right):
                x_offset = left_x2 - left_x1
                y_offset = int(height / 2)
            elif abs(steering_angle_left) > abs(steering_angle_right):
                x_offset = right_x2 - right_x1
                y_offset = int(height / 2)
        else: #normal left line
                mid = int(width / 2)
                x_offset = (left_x2 + right_x2) / 2 - mid
                y_offset = int(height / 2)
    elif len(lane_lines) == 1:
                x1, _, x2, _ = lane_lines[0][0]
                x_offset = x2 - x1
                y_offset = int(height / 2)
    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)     
    #angle_to_mid_radian = math.atan(x_offset / y_offset)
    alfa = 0.6
    angle_to_mid_radian =(1-alfa)*math.atan(x_offset/ y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg +90
    angle = angle_to_mid_radian
    return steering_angle

def display_heading_line(frame, steering_angle, line_color=(0, 255,0), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 1.75)
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image

current_command = None  # Biáº¿n lÆ°u trá»¯ lá»‡nh hiá»‡n táº¡i
def send_command(msgID,angle):
    global current_command
#     if i==0:
#         send_command(1,10)
#         angle=10
    if angle != current_command:  # Kiá»ƒm tra xem lá»‡nh má»›i cÃ³ khÃ¡c vá»›i lá»‡nh hiá»‡n táº¡i hay khÃ´ng
        current_command = angle  # Cáº­p nháº­t lá»‡nh hiá»‡n táº¡i
        # Send the command with the new angle
        command = f"#{msgID}:{current_command};;\r\n"  # Structure of command from RPi
        ser.write(command.encode())
        time.sleep(0.5)  # Delay for Nucleo to response
        response = ser.readline().decode().strip()
        print(f"Response from Nucleo: {response}")
        print(f"#{msgID}:{current_command}")
        # XÃ³a grrent_command}iÃ¡ trá»‹ Ä‘Ã£ gá»­i khá» i hÃ ng Ä‘á»£i# Khá»Ÿi táº¡o webcam
#         i=1
        
def send_command_steering(msgID,steering_angles):
    print(msgID,steering_angles)
    """Send the first steering angle from the queue to Nucleo."""
        # Láº¥y giÃ¡ trá»‹ Ä‘áº§u tiÃªn tá»« hÃ ng Ä‘á»£i mÃ  khÃ´ng xÃ³a nÃ³
    content = steering_angles  # Sá»­ dá»¥ng popleft() Ä‘á»ƒ láº¥y vÃ  xÃ³a.
    if content >25:
       command = f"#{msgID}:{25};;\r\n"  # Structure of command from RPi 
    elif content <-25:
       command = f"#{msgID}:{-25};;\r\n"  # Structure of command from RPi 
    else:
       command = f"#{msgID}:{content};;\r\n"  # Structure of command from RPi
    ser.write(command.encode())
    time.sleep(0.1)  # Delay for Nucleo to response
    response = ser.readline().decode().strip()
    print(f"Response from Nucleo: {response}")
    print(msgID,steering_angles)
    # XÃ³a giÃ¡ trá»‹ Ä‘Ã£ gá»­i khá» i hÃ ng Ä‘á»£i
            
# DETCETION            
            
            
# Object
def detect_objects(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (width_image, height_image))
    input_data = np.expand_dims(image_resized, axis=0)
        
    if float_input:
        input_data = (np.float32(input_data) - 127.5) / 127.5
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
        
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]
    signal='none'
    for i in range(len(scores)):
        if scores[i] > min_confidence and scores[i] <= 1.0:
            ymin = int(max(1, boxes[i][0] * imH))
            xmin = int(max(1, boxes[i][1] * imW))
            ymax = int(min(imH, boxes[i][2] * imH))
            xmax = int(min(imW, boxes[i][3] * imW))
            object_name = labels[int(classes[i])]
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
                    send_command(1,0)
                    time.sleep(2)
    return(signal)






# sign and light
def sign_detection(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stops = stop_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        parkings = parking_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        cross = cross_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        straight = straight_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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
    
def Traffic_Light_detection(frame):
        # Chuyá»ƒn Ä‘á»•i frame sang áº£nh xÃ¡m
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # PhÃ¡t hiá»‡n Ä‘Ã¨n giao thÃ´ng mÃ u Ä‘á» 
        red_lights = Red_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # PhÃ¡t hiá»‡n Ä‘Ã¨n giao thÃ´ng mÃ u vÃ ng
        yellow_lights = Yellow_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # PhÃ¡t hiá»‡n Ä‘Ã¨n giao thÃ´ng mÃ u xanh
        green_lights = Green_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Máº·c Ä‘á»‹nh lÃ  khÃ´ng cÃ³ tÃ­n hiá»‡u
        signal = 'No Signal'
        
        # Kiá»ƒm tra tÃ­n hiá»‡u Ä‘Ã¨n giao thÃ´ng
        if len(red_lights) > 0:
            signal = 'Red Light'
            print("Traffic Light Signal:", signal)
        elif len(yellow_lights) > 0:
            signal = 'Yellow Light'
            print("Traffic Light Signal:", signal)
        elif len(green_lights) > 0:
            signal = 'Green Light'
            print("Traffic Light Signal:", signal)
        
        # In tÃ­n hiá»‡u
        #print("Traffic Light Signal:", signal)
        
        # Váº½ hÃ¬nh chá»¯ nháº­t vÃ  thÃªm vÄƒn báº£n cho má»—i loáº¡i Ä‘Ã¨n giao thÃ´ng Ä‘Æ°á»£c phÃ¡t hiá»‡n
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
    
def run_detection(frame):
     # Process the obstacle detection: 
    signal2=detect_objects(frame)    
    if signal2=='none':
        send_command(1,10)

    # Process the frame (detect signs)
    sign_signal, dist_sign = sign_detection(frame)

    if sign_signal == 'Stops':
        if dist_sign < 15:
            send_command(1, 0)
            time.sleep(5)
            send_command(1, 10)
            time.sleep(2)
            
    elif sign_signal == 'Parkings':
        if dist_sign < 10 and dist_sign > 0:
            send_command(1, 5)
            
    elif sign_signal == 'None':
        send_command(1, 10)  # Example: normal operation
            
    #process the trafficlight detection: 
    light_signal, dist_red_light = Traffic_Light_detection(frame)

    if light_signal == 'Red Light':
        if dist_red_light < 10 and dist_red_light > 5:
            send_command(1, 0)
            time.sleep(5)
            send_command(1, 10)
            time.sleep(2)
            
        elif light_signal == 'Yellow Light':
            send_command(1, 5)
            
        elif light_signal == 'None':
            send_command(1, 10)  # Example: normal operation 
            
            

send_command(1, 15)
# cv2.startWindowThread()
# picam2 = Picamera2()
# # picam2 = cv2.VideoCapture("test_map3.mp4")
# picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
# picam2.start()

height = 704
width = 1279
picam2 = Picamera2()
old_lines = None 
def processing():
    # picam2 = cv2.VideoCapture("test_map3.mp4")
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    while True:
        
        frame = picam2.capture_array()
#         run_detection(frame)

    #     frame = picam2.read() 
        frame_1 =cv2.resize(np.copy(frame) , (width , height))
        canny_image = detect_edges(frame_1)
        cropped_canny = region_of_interest(canny_image)
        lines  = detect_line_segments(cropped_canny)
        averaged_lines = average_slope_intercept(frame_1, lines)
        line_image = display_lines(frame_1, averaged_lines)
        steering_angle = get_steering_angle(frame_1, averaged_lines)
        heading_image = display_heading_line(line_image,steering_angle)
        combo_image = cv2.addWeighted(frame_1, 0.8, heading_image, 1, 1)
        steering_angle = steering_angle - 90
        steering_angles.append(steering_angle)
        # send_command_steering(2,steering_angles[0])
        old_lines = averaged_lines

#         cv2.imshow('result', combo_image)   
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def control_thread():
    while True:
        if steering_angles:
            angle = sum(steering_angles) / len(steering_angles)  # Lấy trung bình góc lái từ hàng đợi
            send_command_to_nucleo(angle)  # Gửi lệnh đến Nucleo
        time.sleep(0.1)
def send_command_steering():
#     print(msgID,steering_angles)
    msgID =2 
    """Send the first steering angle from the queue to Nucleo."""
        # Lấy giá trị đầu tiên từ hàng đợi mà không xóa nó
    while True:
        if steering_angles:
            content = steering_angles[0]
            # Sử dụng popleft() để lấy và xóa.
            if content >25:
               command = f"#{msgID}:{25};;\r\n"  # Structure of command from RPi 
            elif content <-25:
               command = f"#{msgID}:{-25};;\r\n"  # Structure of command from RPi 
            else:
               command = f"#{msgID}:{content};;\r\n"  # Structure of command from RPi
            ser.write(command.encode())
            time.sleep(0.1)  # Delay for Nucleo to response
            response = ser.readline().decode().strip()
            print(f"Response from Nucleo: {response}")
            print(msgID,steering_angles[0])
def record():
    picam2.start_and_record_video("test_map4.mp4", duration=180)
    
    # Xóa giá trị đã gửi khỏi hàng đợi
# Hàm để gửi lệnh đến Nu
# Khởi tạo luồng cho xử lý đối tượng và gửi lệnh
object_detection_thread = threading.Thread(target=processing)
control_thread = threading.Thread(target=send_command_steering)
record_thread = threading.Thread(target=record)
# Khởi động các luồng
object_detection_thread.start()
control_thread.start()
record_thread.start()

# Chờ cho các luồng kết thúc
object_detection_thread.join()
control_thread.join()
record_thread.join()



