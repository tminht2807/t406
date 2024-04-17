import threading
import board
import adafruit_bno055
import serial
import time

class AutonomousCarController(threading.Thread):
    def __init__(self, serial_port='/dev/ttyACM0', baud_rate=19200, yaw_threshold=10, stability_period=5):
        super().__init__()
        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        self.i2c = board.I2C()
        self.sensor = adafruit_bno055.BNO055_I2C(self.i2c)
        self.previous_yaw = None
        self.yaw_threshold = 5
        self.stability_period = stability_period
        self.running = True
        self.low_speed_sent = False
        self.normal_speed_sent = False
        self.high_speed_sent = False

    def send_command(self, msgID, value):
        print(f"Sending command {msgID} with value {value}")
        command = f"#{msgID}:{value};;\r\n"
        self.ser.write(command.encode())
        time.sleep(0.2)
        response = self.ser.readline().decode().strip()
        print(f"Response from Nucleo: {response}")

    def check_yaw_change(self, current_yaw):
        if self.previous_yaw is None:
            self.previous_yaw = current_yaw
            return

        yaw_change = abs(current_yaw - self.previous_yaw)
        if yaw_change > self.yaw_threshold:
            if not self.low_speed_sent:
                self.send_command(1, 8)
                self.low_speed_sent = True
                self.normal_speed_sent = False
                self.high_speed_sent = False
        self.previous_yaw = current_yaw
        return yaw_change
    def handle_pitch(self, pitch):
        if pitch > 5 and not self.high_speed_sent:
            self.send_command(1, 2)
            self.high_speed_sent = True
            self.normal_speed_sent = False
            self.low_speed_sent = False
        elif -2 < pitch < 5 and not self.normal_speed_sent:
            self.send_command(1, 10)
            self.normal_speed_sent = True
            self.high_speed_sent = False
            self.low_speed_sent = False
        elif pitch < 0 and not self.low_speed_sent:
            self.send_command(1, 6)
            self.low_speed_sent = True
            self.normal_speed_sent = False
            self.high_speed_sent = False

    def run(self):
        while self.running:
            x,y,z = self.sensor.acceleration
            yaw, pitch, _ = self.sensor.euler
            yaw_change = self.check_yaw_change(yaw)
            if yaw_change is not None:
                if yaw_change<self.yaw_threshold:
                    self.handle_pitch(pitch)
            time.sleep(1)
            if x< self.x_threshold:
                time.sleep(13)
                self.running = True
                self.low_speed_sent = False
                self.normal_speed_sent = False
                self.high_speed_sent = False
                                

    def stop(self):
        self.running = False
