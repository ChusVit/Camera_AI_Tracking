import cv2
# Import module AI chúng ta vừa viết
from vision.detector_yolo import YoloDetector

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể kết nối với Camera.")
        return

    # Khởi tạo "Bộ não thị giác"
    detector = YoloDetector()

    # Lấy tọa độ tâm của toàn bộ màn hình camera (đây là điểm ta muốn người luôn nằm ở đó)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_center_x = frame_width // 2
    screen_center_y = frame_height // 2

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # --- TẦNG 1: PERCEPTION (Nhận thức) ---
        # Ném khung hình cho AI xử lý, lấy lại khung hình đã vẽ và tọa độ người
        frame, person_x, person_y = detector.detect(frame)

        # Vẽ tâm màn hình (màu xanh dương) để làm điểm mốc chuẩn
        cv2.circle(frame, (screen_center_x, screen_center_y), 5, (255, 0, 0), -1)

        # Nếu phát hiện ra người, tính độ lệch (chuẩn bị cho bước sau)
        if person_x is not None and person_y is not None:
            error_x = person_x - screen_center_x
            error_y = person_y - screen_center_y
            
            # Vẽ đường thẳng nối từ tâm màn hình đến người (đường ngắm)
            cv2.line(frame, (screen_center_x, screen_center_y), (person_x, person_y), (0, 255, 255), 2)
            
            # Hiển thị thông số trên màn hình
            cv2.putText(frame, f"Err X: {error_x} | Err Y: {error_y}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Camera Tracking - Perception Layer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()