import cv2
import numpy as np
import time
import threading

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

    return signal, dist_Light


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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Stop Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        dist_Sign = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    for (x, y, w, h) in parkings:
        # Process parking sign
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Parking Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        dist_Sign = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Sign), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    
    return signal, dist_Sign

# Function to send commands
current_command = None  # Biến lưu trữ lệnh hiện tại

def send_command1(angle):
    global current_command  # Sử dụng biến global
    if angle != current_command:  # Kiểm tra xem lệnh mới có khác với lệnh hiện tại hay không
        current_command = angle  # Cập nhật lệnh hiện tại
        # Send the command with the new angle
        print("Sending command:",current_command)
        # Code to send the command

def send_command(msgID,steering_angles): #stop function and send
  print(msgID,steering_angles)

# Function to capture frames from the camera
def camera_capture():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process the frame (detect signs, lights, etc.)
        sign_signal, dist_sign = Sign_detection(frame)
        light_signal, dist_light = Traffic_Light_detection(frame)
        
        if sign_signal == 'Stops' or light_signal == 'Red Light':
            if dist_sign < 10 and dist_sign > 5:
                send_command(1,0)
                time.sleep(5)
                send_command(1,10)
        
        elif sign_signal == 'Parkings' or light_signal == 'Yellow Light':
            if dist_sign < 10 and dist_sign > 0:
                send_command(1,5)
        
        elif sign_signal == 'None' or light_signal == 'Green Light':
            send_command(1,10)  # Example: normal operation
        
        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Start the camera capture thread
camera_thread = threading.Thread(target=camera_capture)
camera_thread.daemon = True  # Set as daemon to exit when the main program exits
camera_thread.start()

# Join the camera thread to the main program (not necessary as it's a daemon thread)
camera_thread.join()
