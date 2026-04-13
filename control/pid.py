import time

class PIDController:
    def __init__(self, kp, ki, kd):
        """
        Khởi tạo bộ điều khiển PID.
        kp: Hệ số tỉ lệ (Quyết định tốc độ phản hồi)
        ki: Hệ số tích phân (Khử sai số tĩnh khi gần tới đích)
        kd: Hệ số đạo hàm (Kháng lại sự thay đổi, đóng vai trò như phanh giảm chấn)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.prev_error = 0
        self.integral = 0
        self.prev_time = time.time()

    def compute(self, error):
        """
        Tính toán giá trị đầu ra (vận tốc/góc quay) dựa trên độ lệch (error).
        """
        current_time = time.time()
        # Tính thời gian trôi qua giữa 2 frame (delta time)
        dt = current_time - self.prev_time 
        
        # Tránh lỗi chia cho 0 nếu máy chạy quá nhanh
        if dt <= 0.0:
            dt = 0.01

        # 1. Khâu Tỉ lệ (Proportional)
        p_term = self.kp * error

        # 2. Khâu Tích phân (Integral)
        self.integral += error * dt
        # Chống tích lũy quá mức (Anti-windup) - giới hạn I_term để không bị quay lố quá xa
        self.integral = max(min(self.integral, 1000), -1000) 
        i_term = self.ki * self.integral

        # 3. Khâu Đạo hàm (Derivative)
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Cập nhật trạng thái cho vòng lặp tiếp theo
        self.prev_error = error
        self.prev_time = current_time

        # Tổng hợp Tín hiệu điều khiển Output
        output = p_term + i_term + d_term
        return output
    
    def reset(self):
        """Reset các giá trị khi mất dấu mục tiêu (không thấy người)"""
        self.prev_error = 0
        self.integral = 0
        self.prev_time = time.time()