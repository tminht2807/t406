import cv2
import numpy as np
#import serial 
from collections import deque
import time
import threading
#from picamera2 import Picamera2
#ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
steering_angles = deque(maxlen=10)

# Khởi tạo CascadeClassifier và đọc tệp xml
Stop_Cascade = cv2.CascadeClassifier("C:\\Users\\Phong Phung\\Desktop\\duy\\stopcascade.xml")
Parking_Cascade= cv2.CascadeClassifier("C:\\Users\\Phong Phung\\Desktop\\duy\\parking_cascade.xml")

# Khởi tạo CascadeClassifier và đọc tệp xml cho đèn giao thông màu đỏ, vàng và xanh
Red_Light_Cascade = cv2.CascadeClassifier("C:\\Users\\Phong Phung\\Desktop\\testred\\classifier\\cascadered.xml")
Yellow_Light_Cascade = cv2.CascadeClassifier("C:\\Users\\Phong Phung\\Desktop\\testyellow\\classifier\\cascadeyellow.xml")
Green_Light_Cascade = cv2.CascadeClassifier("C:\\Users\\Phong Phung\\Desktop\\testgreen\\classifier\\cascadegreen.xml")


def Traffic_Light_detection(frame):
    # Chuyển đổi frame sang ảnh xám
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện đèn giao thông màu đỏ
    red_lights = Red_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Phát hiện đèn giao thông màu vàng
    yellow_lights = Yellow_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Phát hiện đèn giao thông màu xanh
    green_lights = Green_Light_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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

    return [dist_red, dist_green, dist_yellow]

def Sign_detection(frame):
    # Khai báo các biến
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stops = Stop_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    parkings = Parking_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if stop sign is detected
    dist_Stop = 0
    dist_Parking = 0
    if len(stops) > 0:
        print("Signal: Stop Sign")
    
    # Check if parking sign is detected
    if len(parkings) > 0:
        print("Signal: Parking Sign")
    
    # Vòng lặp qua các biển báo được phát hiện
    for (x, y, w, h) in stops:
        # Vẽ hình chữ nhật và thêm văn bản "Stop Sign"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Stop Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        dist_Stop = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Stop), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

        
    for (x, y, w, h) in parkings:
        # Vẽ hình chữ nhật và thêm văn bản "Parking Sign"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Parking Sign', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        dist_Parking = int((-0.4412)*w+63.9706)
        cv2.putText(frame, 'D = {} cm'.format(dist_Parking), (130,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    return [dist_Stop, dist_Parking]



def send_command(msgID,steering_angles): #stop function and send
    print(msgID,steering_angles)
    time.sleep(0.1)  # Delay for Nucleo to response
'''
    """Send the first steering angle from the queue to Nucleo."""
        # Lấy giá trị đầu tiên từ hàng đợi mà không xóa nó
    content = steering_angles  # Sử dụng popleft() để lấy và xóa.
    command = f"#{msgID}:{content};;\r\n"  # Structure of command from RPi
    ser.write(command.encode())
    time.sleep(0.5)  # Delay for Nucleo to response
    response = ser.readline().decode().strip()
    print(f"Response from Nucleo: {response}")
    # Xóa giá trị đã gửi khỏi hàng đợi# Khởi tạo webcam
    
'''
'''
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
send_command(1,10)
'''
def Stop_car(distance):
     if distance < 10 and distance > 5:
        send_command(1,0)
        time.sleep(5)
        send_command(1,10)
        time.sleep(5)

def Slow_car(distance)
     if distance < 10 and distance > 5:
        send_command(1,5)
        time1 = time.time()
        while(1):
            time2 = time.time()
            if(time2 - time1) >= 5:
                send_command(1,10)
                break

def Slow_car(distance):
    def send_command_thread(msgID, steering_angles):
        send_command(msgID, steering_angles)

    start_time = time.time()
    if 5 < distance < 20:  # Kiểm tra khoảng cách nằm trong khoảng từ 5 đến 20
        while True:
            current_time = time.time() - start_time  # Tính thời gian đã trôi qua
            if current_time < 5:
                send_thread = threading.Thread(target=send_command_thread, args=(1, 5))
                send_thread.start()
                send_thread.join()  # Chờ luồng gửi lệnh hoàn thành trước khi tiếp tục
            elif 5 <= current_time < 10:
                send_thread = threading.Thread(target=send_command_thread, args=(1, 10))
                send_thread.start()
                send_thread.join()  # Chờ luồng gửi lệnh hoàn thành trước khi tiếp tục
            else:
                break  # Kết thúc vòng lặp khi thời gian đã trôi qua vượt quá 10 giây

def send_commands_from_queue(distance):
    while True:
        if steering_angles:
            if 5 < distance < 20:
                send_command(1, 5)
            else:
                send_command(1, steering_angles[0])  # Gửi góc lái đầu tiên trong hàng đợi
                print(steering_angles[0])
                steering_angles.popleft()  # Loại bỏ phần tử đã gửi khỏi hàng đợi

                time.sleep(0.01)  # Đợi 0.5 giây trước khi gửi lệnh tiếp theo
        else:
            time.sleep(0.5)  # Nếu không có phần tử nào trong hàng đợi, chờ một chút trước khi kiểm tra lại

# Khởi tạo và bắt đầu thread
command_thread = threading.Thread(target=send_commands_from_queue)
command_thread.daemon = True  # Đặt là daemon thread để chương trình có thể thoát khi chỉ còn daemon threads
command_thread.start()

cv2.startWindowThread()      
                
        
        		

cap = cv2.VideoCapture(0)
while True:
    # Đọc frame từ webcam
        #frame = picam2.capture_array() 
    ret, frame = cap.read()  
    RoI_detection = frame[0:300, 320:640] # Điều chỉnh kích thước này tùy thuộc vào kích thước frame(check frame của cam )
    
    '''#Đo kích thước khung hình
        height, width, channels = RoI_detection.shape
        print("Kích thước video gốc: {}x{} pixel".format(width, height))'''

    # Gọi hàm Traffic_Light_detection để phát hiện đèn giao thông màu đỏ, vàng và xanh
    #Traffic_Light_detection(RoI_detection) 
    # Gọi hàm Sign_detection để phát hiện biển báo stop và parking
    dist_sign = Sign_detection(RoI_detection) 
    dist_Stop = dist_sign[0]
    dist_Parking = dist_sign[1] 
    Stop_car(dist_Stop)
    #Slow_car(dist_Parking)

    dist_light = Traffic_Light_detection(RoI_detection)
    dist_light_R=dist_light[0]
    dist_light_Y=dist_light[1]
    dist_light_G=dist_light[2]
    Stop_car(dist_light_R)
    


    # Hiển thị frame
    cv2.imshow('Traffic Light and Sign Detection', RoI_detection)   
    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
#cap.release()
cv2.destroyAllWindows()
