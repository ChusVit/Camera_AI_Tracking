from ultralytics import YOLO
import cv2
import random

class YoloDetector:
    def __init__(self, model_name='yolov8n-pose.pt'):
        print("Đang tải mô hình Pose AI...")
        self.model = YOLO(model_name)
        
        # Bộ lọc làm mượt cho người được target
        self.smooth_x = None
        self.smooth_y = None
        self.alpha = 0.4 
        
        # Từ điển lưu trữ màu sắc cho từng ID người {ID: (B, G, R)}
        self.id_colors = {}

        # --- Định nghĩa Skeleton Connections (Cặp điểm xương chuẩn của YOLO) ---
        # 17 Keypoints: [Mũi, Mắt T, Mắt P, Tai T, Tai P, Vai T, Vai P, Khuỷu tay T, Wrist T, ...]
        self.SKELETON_CONNECTIONS = [
            # Đầu
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Thân
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Tay T
            (5, 7), (7, 9),
            # Tay P
            (6, 8), (8, 10),
            # Chân T
            (11, 13), (13, 15),
            # Chân P
            (12, 14), (14, 16)
        ]
        
        print("Tải mô hình thành công!")

    def get_color_for_id(self, track_id):
        """Hàm sinh màu ngẫu nhiên nhưng cố định cho một ID"""
        if track_id not in self.id_colors:
            # Random màu RGB
            self.id_colors[track_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        return self.id_colors[track_id]

    def detect(self, frame):
        results = self.model.track(frame, persist=True, classes=0, conf=0.6, verbose=False)
        
        target_box = None
        max_area = 0 

        # Kiểm tra xem có track được ai và có ID không
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Lấy thông tin Bounding Box, ID và Keypoints (Điểm xương)
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.int().cpu().tolist()

            for box, track_id, kpts in zip(boxes, track_ids, keypoints):
                x1, y1, x2, y2 = box
                color = self.get_color_for_id(track_id)
                
                # --- Vẽ Bounding Box và ID cho từng người ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- NÂNG CẤP: Vẽ Khung Xương (Skeleton Wireframe) ---
                # 1. Vẽ các điểm khớp xương trước
                for kp in kpts:
                    kx, ky = kp
                    if kx != 0 and ky != 0: 
                        cv2.circle(frame, (kx, ky), 4, color, -1) 

                # 2. Vẽ các đường nối giữa các điểm khớp xương
                for pair in self.SKELETON_CONNECTIONS:
                    kp_a_idx, kp_b_idx = pair
                    kp_a = kpts[kp_a_idx]
                    kp_b = kpts[kp_b_idx]
                    
                    # Nếu cả hai điểm trong cặp đều được AI nhìn thấy
                    if kp_a[0] != 0 and kp_a[1] != 0 and kp_b[0] != 0 and kp_b[1] != 0:
                        cv2.line(frame, (kp_a[0], kp_a[1]), (kp_b[0], kp_b[1]), color, 2)

                # Logic Target: Vẫn tìm người to nhất (gần nhất) để bám theo
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_box = (x1, y1, x2, y2)

        # Xử lý làm mượt cho người gần nhất (Target chính)
        if target_box:
            x1, y1, x2, y2 = target_box
            raw_cx = (x1 + x2) // 2
            raw_cy = (y1 + y2) // 2

            if self.smooth_x is None:
                self.smooth_x, self.smooth_y = raw_cx, raw_cy
            else:
                self.smooth_x = int(self.alpha * raw_cx + (1 - self.alpha) * self.smooth_x)
                self.smooth_y = int(self.alpha * raw_cy + (1 - self.alpha) * self.smooth_y)

            # Vẽ điểm Target (Màu đỏ)
            cv2.circle(frame, (self.smooth_x, self.smooth_y), 6, (0, 0, 255), -1)

            return frame, self.smooth_x, self.smooth_y
        else:
            self.smooth_x = None
            self.smooth_y = None
            return frame, None, None

    def check_fall(self, box, kpts):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Tránh lỗi chia cho 0
        if height == 0: return False 
        
        # Nếu chiều rộng lớn hơn chiều cao 1.2 lần -> Khả năng cao đang nằm/té
        if (width / height) > 1.2:
            return True
            
        return False