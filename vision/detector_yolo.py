from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """Khởi tạo mô hình YOLO. Mặc định dùng bản Nano (n) nhẹ nhất"""
        print("Đang tải mô hình AI...")
        self.model = YOLO(model_name)
        print("Tải mô hình thành công!")

    def detect(self, frame):
        """
        Nhận vào 1 khung hình, tìm người, vẽ khung và trả về tọa độ tâm.
        """
        # Chạy nhận diện, classes=0 là chỉ tìm 'person', verbose=False để tắt log rác ở terminal
        results = self.model(frame, classes=0, verbose=False)
        
        person_center_x, person_center_y = None, None
        
        # Biến lưu diện tích lớn nhất để khóa mục tiêu vào người đứng gần camera nhất
        max_area = 0 

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ góc của bounding box
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                
                # Tính diện tích để tìm người to nhất (gần nhất)
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    person_center_x = (x1 + x2) // 2
                    person_center_y = (y1 + y2) // 2
                    
                    # Vẽ Bounding Box quanh người
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Vẽ điểm hồng tâm trên người
                    cv2.circle(frame, (person_center_x, person_center_y), 5, (0, 0, 255), -1)

        return frame, person_center_x, person_center_y