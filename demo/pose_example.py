from app.detectors.pose_detector import PoseDetector
from app.utils.camera_utils import CameraUtils
import cv2


def main():
    detector = PoseDetector()
    cap = CameraUtils.initialize_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = CameraUtils.process_frame(frame)
        processed_frame = detector.process_frame(frame)

        if CameraUtils.display_frame("Pose Detection", processed_frame) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
