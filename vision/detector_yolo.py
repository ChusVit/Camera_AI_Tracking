from ultralytics import YOLO
import cv2
import random
import math
from collections import deque         
import time                           

class YoloDetector:
    def __init__(self, model_name='yolov8n-pose.pt'):
        print("Đang tải mô hình Pose AI...")
        self.model = YOLO(model_name)
        
        self.smooth_x = None
        self.smooth_y = None
        self.alpha = 0.4
        
        self.last_target_box = None
        self.last_target_kpts = None
        self.target_kpts = None

        self.id_colors = {}

        self.SKELETON_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]

        # Bộ đệm lịch sử cho phát hiện té ngã
        # Lưu tuple (timestamp, angle_deg, height, is_fallen_candidate)
        self.fall_history = deque(maxlen=15)  # ~0.5 giây nếu 30fps
        self.fall_confirmed = False
        self.fall_confirmed_frames = 0
        self.FALL_CONFIRM_DURATION = 10       # số frame liên tiếp phải “nằm” để xác nhận
        self.FALL_ANGLE_THRESHOLD = 60        # góc nghiêng > 60° coi là nằm (co the tuy chinh)
        self.FALL_SPEED_THRESHOLD = 120       # độ/giây (thay đổi góc tối thiểu để coi là đột ngột)
        self.FALL_HEIGHT_CHANGE_RATIO = 0.25  # chiều cao giảm > 25% trong 0.5s coi là đột ngột

        print("Tải mô hình thành công!")

    def get_color_for_id(self, track_id):
        if track_id not in self.id_colors:
            self.id_colors[track_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        return self.id_colors[track_id]

    def detect(self, frame):
        results = self.model.track(frame, persist=True, classes=0, conf=0.6, verbose=False)
        
        target_box = None
        max_area = 0
        self.target_kpts = None

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.int().cpu().tolist()

            for box, track_id, kpts in zip(boxes, track_ids, keypoints):
                x1, y1, x2, y2 = box
                color = self.get_color_for_id(track_id)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                for kp in kpts:
                    kx, ky = kp
                    if kx != 0 and ky != 0:
                        cv2.circle(frame, (kx, ky), 4, color, -1)

                for pair in self.SKELETON_CONNECTIONS:
                    kp_a_idx, kp_b_idx = pair
                    kp_a = kpts[kp_a_idx]
                    kp_b = kpts[kp_b_idx]
                    if kp_a[0] != 0 and kp_a[1] != 0 and kp_b[0] != 0 and kp_b[1] != 0:
                        cv2.line(frame, (kp_a[0], kp_a[1]), (kp_b[0], kp_b[1]), color, 2)

                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_box = (x1, y1, x2, y2)
                    self.target_kpts = kpts

        if target_box:
            x1, y1, x2, y2 = target_box
            raw_cx = (x1 + x2) // 2
            raw_cy = (y1 + y2) // 2

            if self.smooth_x is None:
                self.smooth_x, self.smooth_y = raw_cx, raw_cy
            else:
                self.smooth_x = int(self.alpha * raw_cx + (1 - self.alpha) * self.smooth_x)
                self.smooth_y = int(self.alpha * raw_cy + (1 - self.alpha) * self.smooth_y)

            cv2.circle(frame, (self.smooth_x, self.smooth_y), 6, (0, 0, 255), -1)

            self.last_target_box = target_box
            self.last_target_kpts = self.target_kpts

            return frame, self.smooth_x, self.smooth_y
        else:
            self.smooth_x = None
            self.smooth_y = None
            self.last_target_box = None
            self.last_target_kpts = None
            return frame, None, None

    def get_last_target_data(self):
        return self.last_target_box, self.last_target_kpts

    def check_fall_advanced(self, box, kpts):
        """
        Phát hiện té ngã dựa trên phân tích động (multi-frame).
        Trả về True nếu phát hiện té ngã thực sự.
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if height == 0:
            return False

        # --- Tính góc nghiêng thân người ---
        angle_deg = None
        try:
            shoulder_l = kpts[5]
            shoulder_r = kpts[6]
            hip_l = kpts[11]
            hip_r = kpts[12]
            if all(k[0] > 0 and k[1] > 0 for k in [shoulder_l, shoulder_r, hip_l, hip_r]):
                scx = (shoulder_l[0] + shoulder_r[0]) // 2
                scy = (shoulder_l[1] + shoulder_r[1]) // 2
                hcx = (hip_l[0] + hip_r[0]) // 2
                hcy = (hip_l[1] + hip_r[1]) // 2
                dx = hcx - scx
                dy = hcy - scy
                if dy > 10:                     # giảm ngưỡng để nhạy với người xa
                    angle_rad = abs(math.atan(dx / dy))
                    angle_deg = math.degrees(angle_rad)
        except IndexError:
            pass

        current_time = time.time()
        # Thêm mẫu hiện tại vào lịch sử
        self.fall_history.append((current_time, angle_deg, height, False))

        # --- Phân tích lịch sử để tìm sự kiện đột ngột ---
        if len(self.fall_history) < 2:
            return self.fall_confirmed   # chưa đủ dữ liệu

        # Xét trong khoảng thời gian 0.5 giây gần nhất
        recent = [h for h in self.fall_history if current_time - h[0] <= 0.5]
        if len(recent) < 2:
            return self.fall_confirmed

        oldest = recent[0]
        newest = recent[-1]

        # 1. Phát hiện thay đổi góc nhanh
        speed = 0
        if oldest[1] is not None and newest[1] is not None:
            delta_angle = abs(newest[1] - oldest[1])
            delta_time = newest[0] - oldest[0]
            if delta_time > 0:
                speed = delta_angle / delta_time   # độ/giây

        # 2. Phát hiện chiều cao giảm đột ngột
        height_change_ratio = 0
        if oldest[2] > 0:
            height_change_ratio = (oldest[2] - newest[2]) / oldest[2]

        # Nếu tốc độ góc vượt ngưỡng HOẶC chiều cao giảm đột ngột → nghi ngờ té
        sudden_fall = (speed > self.FALL_SPEED_THRESHOLD) or (height_change_ratio > self.FALL_HEIGHT_CHANGE_RATIO)

        # --- Logic xác nhận té (duy trì góc lớn sau sự kiện) ---
        if sudden_fall:
            self.fall_confirmed = True
            self.fall_confirmed_frames = 0   # bắt đầu đếm frame nằm

        if self.fall_confirmed:
            # Kiểm tra xem có đang nằm không (góc lớn hoặc tỉ lệ W/H lớn)
            lying = False
            if angle_deg is not None and angle_deg > self.FALL_ANGLE_THRESHOLD:
                lying = True
            elif width / height > 1.3:        # fallback khi không có keypoint
                lying = True

            if lying:
                self.fall_confirmed_frames += 1
            else:
                # Nếu đứng dậy giữa chừng, giảm bộ đếm (nhưng không reset ngay)
                self.fall_confirmed_frames = max(0, self.fall_confirmed_frames - 1)

            # Nếu nằm đủ lâu → xác nhận té ngã
            if self.fall_confirmed_frames >= self.FALL_CONFIRM_DURATION:
                return True
            # Nếu quá lâu không nằm (ví dụ 2 giây) → reset
            if self.fall_confirmed_frames == 0 and len(self.fall_history) > 30:
                self.fall_confirmed = False

        return False