import cv2
import numpy as np
import serial 
import time
from picamera2 import Picamera2
ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)

# Khởi tạo CascadeClassifier và đọc tệp xml
Stop_Cascade = cv2.CascadeClassifier("/home/admin/Brain/stop_sign2.xml")
Parking_Cascade= cv2.CascadeClassifier("/home/admin/Downloads/parking_cascade.xml")

# Khởi tạo CascadeClassifier và đọc tệp xml cho đèn giao thông màu đỏ, vàng và xanh
Red_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/traffic_sign/cascadered.xml")
Yellow_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/traffic_sign/cascadeyellow.xml")
Green_Light_Cascade = cv2.CascadeClassifier("/home/admin/Downloads/traffic_sign/cascadegreen.xml")

'''
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
    elif len(yellow_lights) > 0:
        signal = 'Yellow Light'
    elif len(green_lights) > 0:
        signal = 'Green Light'
    
    # In tín hiệu
    print("Traffic Light Signal:", signal)
    
    # Vẽ hình chữ nhật và thêm văn bản cho mỗi loại đèn giao thông được phát hiện
    for (x, y, w, h) in red_lights:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Red Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        
    for (x, y, w, h) in yellow_lights:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, 'Yellow Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

    for (x, y, w, h) in green_lights:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Green Light', (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
'''
def Sign_detection(frame):
    # Khai báo các biến
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stops = Stop_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    parkings = Parking_Cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if stop sign is detected
    dist_Stop = 0
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
    return dist_Stop


# ser = serial.Serial('COM4', 19200, timeout=1)
def send_command(msgID,steering_angles):
    """Send the first steering angle from the queue to Nucleo."""
        # Lấy giá trị đầu tiên từ hàng đợi mà không xóa nó
    content = steering_angles  # Sử dụng popleft() để lấy và xóa.
    command = f"#{msgID}:{content};;\r\n"  # Structure of command from RPi
    ser.write(command.encode())
    time.sleep(0.5)  # Delay for Nucleo to response
    response = ser.readline().decode().strip()
    print(f"Response from Nucleo: {response}")
    # Xóa giá trị đã gửi khỏi hàng đợi# Khởi tạo webcam
    
cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
send_command(1,10)

while True:
    # Đọc frame từ webcam
    frame = picam2.capture_array() 
    RoI_detection = frame[0:300, 320:640] # Điều chỉnh kích thước này tùy thuộc vào kích thước frame(check frame của cam )
    #Đo kích thước khung hình
    height, width, channels = RoI_detection.shape
    print("Kích thước video gốc: {}x{} pixel".format(width, height))
    # Gọi hàm Traffic_Light_detection để phát hiện đèn giao thông màu đỏ, vàng và xanh
    #Traffic_Light_detection(RoI_detection) 
    # Gọi hàm Sign_detection để phát hiện biển báo stop và parking
    dist_Stop = Sign_detection(RoI_detection)  
    if dist_Stop < 22 and dist_Stop > 5:
        send_command(1,0)
        time.sleep(5)			# lat nua fix lai ham chuan cua no
        send_command(1,10)
        time.sleep(2)

    # Hiển thị frame
    cv2.imshow('Traffic Light and Sign Detection', RoI_detection)   
    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
