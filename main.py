import cv2
import time                                   # >>> THÊM MỚI
from vision.detector_yolo import YoloDetector
from control.pid import PIDController

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Lỗi: Không thể kết nối với Camera.")
        return

    # 1. Khởi tạo Tầng Nhận thức
    detector = YoloDetector()

    # 2. Khởi tạo Tầng Điều khiển (2 bộ PID độc lập)
    pid_x = PIDController(kp=0.05, ki=0.001, kd=0.01)
    pid_y = PIDController(kp=0.05, ki=0.001, kd=0.01)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    screen_center_x = frame_width // 2
    screen_center_y = frame_height // 2

    # Biến quản lý trạng thái cảnh báo té ngã
    fall_alert_active = False
    last_fall_alert_time = 0
    FALL_ALERT_COOLDOWN = 10  # giây (chỉ báo lại sau 10s)


    print("Hệ thống đã sẵn sàng!")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # --- TẦNG 1: PERCEPTION ---
        frame, person_x, person_y = detector.detect(frame)
        cv2.circle(frame, (screen_center_x, screen_center_y), 5, (255, 0, 0), -1)

        # --- TẦNG 2: CONTROL LOGIC ---
        if person_x is not None and person_y is not None:
            error_x = person_x - screen_center_x
            error_y = person_y - screen_center_y
            
            cv2.line(frame, (screen_center_x, screen_center_y), (person_x, person_y), (0, 255, 255), 2)
            
            # Tích hợp phát hiện té ngã
            target_box, target_kpts = detector.get_last_target_data()
            if target_box and target_kpts:
                is_fallen = detector.check_fall_advanced(target_box, target_kpts)
                if is_fallen:
                    current_time = time.time()
                    if (not fall_alert_active) and (current_time - last_fall_alert_time > FALL_ALERT_COOLDOWN):
                        print("!!! CẢNH BÁO: PHÁT HIỆN TÉ NGÃ !!!")
                        # TODO: Gửi lệnh đến phần cứng (còi, đèn) hoặc tin nhắn Telegram/IFTTT
                        fall_alert_active = True
                        last_fall_alert_time = current_time
                    
                    # Hiển thị cảnh báo 
                    cv2.putText(frame, "FALL DETECTED!", (frame_width//2 - 150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                else:
                    fall_alert_active = False  
            

            # 2.2 Đưa độ lệch vào PID để tính toán tín hiệu xuất
            pan_signal = pid_x.compute(error_x)
            tilt_signal = pid_y.compute(error_y)
            
            # --- TẦNG 3: GIAO TIẾP (Mô phỏng hiển thị trên màn hình) ---
            cv2.putText(frame, f"Err X: {error_x:4d} | Pan CMD: {pan_signal:6.2f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Err Y: {error_y:4d} | Tilt CMD: {tilt_signal:6.2f}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            pid_x.reset()
            pid_y.reset()
            cv2.putText(frame, "Target Lost - Waiting...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Camera Tracking - System Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()