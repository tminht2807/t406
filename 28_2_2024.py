import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import serial
import time
import threading
# Define global variables and queues
# ser = serial.Serial('COM3', 19200, timeout=1)
steering_angles = deque(maxlen=10)
speed = deque(maxlen=10)

# Define CascadeClassifier for traffic signs and lights
Stop_Cascade = cv2.CascadeClassifier("D:\\OKhe\\bosch\\Traffic_sign\\weekend\\stop_sign2.xml")
Parking_Cascade = cv2.CascadeClassifier("D:\\OKhe\\bosch\\Traffic_sign\\weekend\\Cascade_Parking\\parking_cascade.xml")
Red_Light_Cascade = cv2.CascadeClassifier("D:\\OKhe\\bosch\\Traffic_light\\Light_cascade\\cascadered.xml")
Yellow_Light_Cascade = cv2.CascadeClassifier("D:\\OKhe\\bosch\\Traffic_light\\Light_cascade\\cascadeyellow.xml")
Green_Light_Cascade = cv2.CascadeClassifier("D:\\OKhe\\bosch\\Traffic_light\\Light_cascade\\cascadegreen.xml")




# Function to detect traffic lights
def Traffic_Light_detection(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect red lights
    red_lights = Red_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Detect yellow lights
    yellow_lights = Yellow_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Detect green lights
    green_lights = Green_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Default signal
    signal = 'None'

    # Check traffic light signals
    if len(red_lights) > 0:
        signal = 'Red Light'
        print("Traffic Light Signal:", signal)
    elif len(yellow_lights) > 0:
        signal = 'Yellow Light'
        print("Traffic Light Signal:", signal)
    elif len(green_lights) > 0:
        signal = 'Green Light'
        print("Traffic Light Signal:", signal)

    # Vẽ hình chữ nhật và thêm văn bản cho mỗi loại đèn giao thông được phát hiện
    dist_Light=0
    for (x, y, w, h) in red_lights:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Red Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        dist_Light = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Light), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    for (x, y, w, h) in yellow_lights:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, 'Yellow Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
        dist_Light = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Light), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    for (x, y, w, h) in green_lights:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Green Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        dist_Light = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Light), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    #return [dist_red, dist_green, dist_yellow, signal]
    return [signal, dist_Light] 


# Function to detect traffic signs
def Sign_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stops = Stop_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    parkings = Parking_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    signal = 'None'
    dist_Sign=0
    # Check traffic light signals
    if len(stops) > 0:
        signal = 'Stops'
        print("Sign Signal:", signal)
    elif len(parkings) > 0:
        signal = 'Parkings'
        print("Sign Signal:", signal)

    for (x, y, w, h) in stops:
        # Process stop sign
         # Vẽ hình chữ nhật và thêm văn bản "Stop Sign"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Stop Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        dist_Sign = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    for (x, y, w, h) in parkings:
        # Process parking sign
        # Vẽ hình chữ nhật và thêm văn bản "Parking Sign"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Parking Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        dist_Sign = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    #return [dist_Stop, dist_Parking]
    return [signal, dist_Sign]



#Nam-----------------------------------------------------------
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # Đảm bảo line là mảng 1 chiều có 4 phần tử
            if not isinstance(line, np.ndarray):
                line = np.array(line)
            if line.ndim == 2 and line.shape[0] == 1:
                line = line.flatten()
            
            # Kiểm tra xem line có đủ 4 phần tử không
            if len(line) == 4:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            else:
                print("Đoạn thẳng không hợp lệ:", line)

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
    # reduce the noise of the image: if it’s not between the 5-by-5 Kernel it is deleted
    edges = cv2.Canny(blur, 100, 200) #100 and 200 are min and max gradient intensities
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




'''
def send_command(speed,steering_angles):
    global current_command  # Sử dụng biến global
    while True:
        if speed:
            new_command = speed[0]  # Lấy lệnh tiếp theo từ hàng đợi
            if new_command != current_command:  # Kiểm tra xem lệnh mới có khác với lệnh hiện tại hay không
                current_command = new_command  # Cập nhật lệnh hiện tại
                # Send the command with the new angle
                print("Sending command:",(current_command))
                # Code to send the command
        if steering_angles: 
            steering_command = steering_angles[0]
            print("Sending command:",(steering_command))


        time.sleep(0.1)  # Delay for Nucleo to respond
'''
# Function to send commands

def send_command():
    while True:
        if speed:
            # Pop the first steering angle from the queue
            tocdo = speed[0]
            speed.popleft()    
            # Send the command with the angle
            print("Sending command speed:", tocdo)
            # Code to send the command
        time.sleep(0.1)  # Delay for Nucleo to respond

        if steering_angles: 
            angle = steering_angles[0]
            steering_angles.popleft()

            print("Sending command steering:", angle)
        time.sleep(0.1)  # Delay for Nucleo to respond


        
#def send_command(msgID,steering_angles):
    #print(msgID,steering_angles)

    '''
    global current_command  # Sử dụng biến global
    while True:
        if speed:
            new_speed = speed[0]  # Lấy tốc độ tiếp theo từ hàng đợi
            if new_speed != current_command:  # Kiểm tra xem tốc độ mới có khác với tốc độ hiện tại hay không
                current_command = new_speed  # Cập nhật tốc độ hiện tại
                print("Sending speed command:", current_command)
                # Code to send the speed command
                # Example: ser.write(current_command.encode())
        if steering_angles:
            print("Sending steering angle command:", steering_angles[0])
                # Code to send the steering angle command
                # Example: ser.write(current_command.encode())

        time.sleep(0.1)  # Delay for Nucleo to respond
    '''



def send_signal(signal_, dist_):
    determine_signal = 'Nothing'
    if signal_ == 'Stops'or signal_ == 'Red Light':
        if dist_ < 10 and dist_ > 5:
            determine_signal = 'Stop'
    elif signal_ == 'Parkings' or signal_ == 'Yellow Light':
        if dist_ < 10 and dist_ > 0:
            determine_signal = 'Slow'
        #trả về tốc độ ban đầu
    elif signal_ == 'None' or signal_ == 'Green Light':
        determine_signal = 'Back' 
    
    return determine_signal


def load_image(image_path):
    # Load an image from the specified file
    return cv2.imread(image_path)

# Function to capture frames from the camera
def camera_capture():
    cap = cv2.VideoCapture(0)
    height = 704
    width = 1279
    old_lines = None 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        frame_lane =cv2.resize(np.copy(frame) , (width , height))
        canny_image = detect_edges(frame)
        cropped_canny = region_of_interest(canny_image)
        lines  = detect_line_segments(cropped_canny)
        try:
            averaged_lines = average_slope_intercept(frame_lane, lines)
            line_image = display_lines(frame_lane, averaged_lines)
            steering_angle = get_steering_angle(frame_lane, averaged_lines)
            heading_image = display_heading_line(line_image,steering_angle)
            combo_image = cv2.addWeighted(frame_lane, 0.8, heading_image, 1, 1)
            steering_angle = steering_angle - 90
            steering_angles.append(steering_angle)
            #send_command(2,steering_angles[0])  # Sử dụng ID 2 cho góc lái 
            #time.sleep(0.5)
            # if steering_angles:
            #     send_command(2, steering_angles[0])  # Gửi góc lái đầu tiên trong hàng đợi
                # steering_angles.popleft()  # Loại bỏ phần tử đã gửi khỏi hàng đợi
                # time.sleep(0.5)
            old_lines = averaged_lines
            cv2.imshow('result', combo_image)
        except:
            if old_lines is not None:
                line_image = display_lines(frame_lane, old_lines)
                combo_image = cv2.addWeighted(frame_lane, 0.3, heading_image, 1, 1)
                steering_angle = 90
            cv2.imshow('result', combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
       
        #đoạn detection

        # Process the frame (detect signs, lights, etc.)
        Sign_output=Sign_detection(frame)
        signal_light = Traffic_Light_detection(frame)
        signal_sign = Sign_output[0]
        dist_sign = Sign_output[1]
        # Placeholder for decision making based on detected signals
        signal=send_signal(signal_sign,dist_sign)
    
   
        if signal == 'Stop':
            speed.append(0)
            time1 = time.time()
            while True:
                time2 = time.time()
                if time2 - time1 >= 5:
                    speed.append(10)
                    time.sleep(3)
                    '''
                    if signal == 'Stop':
                        steering_angles.append(10)  # Thêm dòng này
                        break
                    '''
                    break

    
        elif signal=='Slow':
            speed.append(5)
        #trả về tốc độ ban đầu
        elif signal == 'Back':
            speed.append(10)  # Example: normal operation
        
        
    


        
        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
    return(frame)

# Start threads for camera capture and command sending
camera_thread = threading.Thread(target=camera_capture)
command_thread = threading.Thread(target=send_command)

# Set threads as daemon so they exit when the main program exits
camera_thread.daemon = True
command_thread.daemon = True



# Start threads
camera_thread.start()
command_thread.start()


# Join threads to main program (not really necessary since they are daemons)
camera_thread.join()
command_thread.join()



    