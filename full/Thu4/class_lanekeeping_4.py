import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import serial
import time
import threading

class LaneDetectionThread(threading.Thread):
    def __init__(self,cap):
        super().__init__()
        self.steering_angles = deque(maxlen=10)
        self.cap = cap
        self.ser = serial.Serial('/dev/ttyACM0', 19200, timeout=1)
        self.height = 704
        self.width = 1279
        self.old_lines = None
        self.errors = []


    def run(self):
#         cv2.startWindowThread()
        height = 704
        width = 1279
        old_lines = None
#         self.send_command(1,15)
        while True:
#            _,lane_line = self.cap.read()
            lane_line = self.cap.capture_array()
            frame = cv2.resize(np.copy(lane_line) , (width , height))
            canny_image = self.detect_edges(frame)
            cropped_canny = self.region_of_interest(canny_image)
            lines = self.detect_line_segments(cropped_canny)
            try:
                averaged_lines = self.average_slope_intercept(frame, lines)
                line_image = self.display_lines(frame, averaged_lines)
                steering_angle = self.get_steering_angle(frame, averaged_lines)
                heading_image = self.display_heading_line(line_image, steering_angle)
                combo_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
                steering_angle = steering_angle - 90
#                 print(steering_angle) 
                self.steering_angles.append(steering_angle)
                old_lines = averaged_lines
#                 cv2.imshow('result', combo_image)
                self.send_command(2,self.steering_angles[0])  # Gửi lệnh lái
            except:
                if old_lines is not None:
                    line_image = self.display_lines(frame, old_lines)
                    combo_image = cv2.addWeighted(frame, 0.3, heading_image, 1, 1)
                    steering_angle = 90
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

        cv2.destroyAllWindows()
    
    
        # Release the capture
 


    def load_image(self, image_path):
        return cv2.imread(image_path)

    def display_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                if not isinstance(line, np.ndarray):
                    line = np.array(line)
                if line.ndim == 2 and line.shape[0] == 1:
                    line = line.flatten()
                if len(line) == 4:
                    x1, y1, x2, y2 = line
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                else:
                    print("Invalid line segment:", line)
        return line_image


    def average_slope_intercept(self, frame, line_segments):
        lane_lines = []
        if line_segments is None:
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []
        boundary = 1/3
        left_region_boundary = width * 2/3
        right_region_boundary = width * 2/3

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0 and slope < -0.3:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                elif slope>0 and slope>0.3:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            lane_lines.append(self.make_points(frame, left_fit_average))

        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            lane_lines.append(self.make_points(frame, right_fit_average))

        return lane_lines

    def make_points(self, frame, line):       
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height
        y2 = int(y1 * 1 / 2)
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]
#     def nothing(x):
#         pass
#         cv2.namedWindow("Trackbars")
# 
#         cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
#         cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
#         cv2.createTrackbar("L - V", "Trackbars", 170, 255, nothing)
#         cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
#         cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
#         cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)    
    
    def detect_edges(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = 7
        blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        # reduce the noise of the image: if it’s not between the 5-by-5 Kernel it is deleted
        edges = cv2.Canny(blur, 100, 200) #100 and 200 are min and max gradient intensities
        return edges
#     def detect_edges(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         kernel = 5
#         blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
#         edges = cv2.Canny(blur, 100, 220)
#         return edges

    def region_of_interest(self, canny):
        height = canny.shape[0]
        width = canny.shape[1]
        mask = np.zeros_like(canny)
        shape = np.array([[(50, height), (width-50, height), (width-50, 300), (50,300)]], np.int32)
        cv2.fillPoly(mask, shape, 255)
        masked_image = cv2.bitwise_and(canny, mask)
#         cv2.imshow("Canny mainlane", masked_image)
        cv2.waitKey(1)
        return masked_image

    def detect_line_segments(self, cropped_edges):
        rho = 1
        theta = np.pi / 180
        min_threshold = 20
        line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, np.array([]), minLineLength=60, maxLineGap=200)
        return line_segments

    def get_steering_angle(self, frame, lane_lines):
        height, width, _ = frame.shape
        if len(lane_lines) == 2:
            left_x1, left_y1, left_x2, left_y2 = lane_lines[0][0]
            right_x1, right_y1, right_x2, right_y2 = lane_lines[1][0]
            slope_l = math.atan((left_x1-left_x2) / (left_y1-left_y2))
            slope_r = math.atan((right_x1-right_x2) / (right_y1-right_y2))
            slope_ldeg = int(slope_l * 180.0 / math.pi)
            steering_angle_left = slope_ldeg  
            slope_rdeg = int(slope_r * 180.0 / math.pi)
            steering_angle_right = slope_rdeg
            if left_x2 > right_x2:
                if abs(steering_angle_left) <= abs(steering_angle_right):
                    x_offset = left_x2 - left_x1
                    y_offset = int(height / 2)
                elif abs(steering_angle_left) > abs(steering_angle_right):
                    x_offset = right_x2 - right_x1
                    y_offset = int(height / 2)
            else:
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
        alfa = 0.6
        angle_to_mid_radian =(1-alfa)*math.atan(x_offset/ y_offset)
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
        steering_angle = angle_to_mid_deg + 90
        return steering_angle

    def display_heading_line(self, frame, steering_angle, line_color=(0, 255, 0), line_width=5):
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
    def send_command(self,msgID,steering_angles):
         # Lấy giá trị đầu tiên từ hàng đợi mà không xóa nó
#          content = steering_angles  # Sử dụng popleft() để lấy và xóa.
         command = f"#{msgID}:{steering_angles};;\r\n"
         if steering_angles >25:
            command = f"#{msgID}:{25};;\r\n"  # Structure of command from RPi 
         elif steering_angles <-25:
            command = f"#{msgID}:{-25};;\r\n"  # Structure of command from RPi 
         else:
            command = f"#{msgID}:{steering_angles};;\r\n"
           
         self.ser.write(command.encode())
          # Delay for Nucleo to response
         response = self.ser.readline().decode().strip()
         print(msgID,steering_angles)
         print(f"Response from Nucleo: {response}")
    def smooth_input(self,angle):
        # Apply a simple moving average filter to smooth the steering angle
        window_size = 2
        if len(self.errors) >= window_size:
            smoothed_angle = sum(self.errors[-window_size:]) / window_size
        else:
            smoothed_angle = angle
        return smoothed_angle
#      
        #  Xóa giá trị đã gửi khỏi hàng đợi
#     def send_command(self,msgID, steering_angles):
#         # Lấy giá trị đầu tiên từ hàng đợi mà không xóa nó
#         content = steering_angles  # Sử dụng popleft() để lấy và xóa.
#         command = f"#{msgID}:{content};;\r\n"  # Structure of command from RPi
#         self.ser.write(command.encode())
#         time.sleep(0.1)  # Delay for Nucleo to response
#         response = self.ser.readline().decode().strip()
#         print(f"Response from Nucleo: {response}")
        # Xóa giá trị đã gửi khỏi hàng đợi

#     def send_commands_from_queue(self):
#         while True:
#             if self.steering_angles:
#                 if self.steering_angles[0]>25 :
#                     self.send_command(2, 25)
#                 if self.steering_angles[0]<-25 :
#                     self.send_command(2, -25)
#                 else:
#                     self.send_command(2, self.steering_angles[0])  # Gửi góc lái đầu tiên trong hàng đợi
#                     self.steering_angles.popleft()  # Loại bỏ phần tử đã gửi khỏi hàng đợi
# 
#                     time.sleep(0.5)  # Đợi 0.5 giây trước khi gửi lệnh tiếp theo
#             else:
#                 time.sleep(0.5)
'''
def main():
    lane_detection_thread = LaneDetectionThread()
    lane_detection_thread.start()
    lane_detection_thread.join()

if __name__ == "__main__":
    main()
'''