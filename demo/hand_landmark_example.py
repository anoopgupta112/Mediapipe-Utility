from app.detectors.hand_landmark_detector import HandLandmarkDetector
from app.utils.camera_utils import CameraUtils
import cv2


def main():
    detector = HandLandmarkDetector()
    cap = CameraUtils.initialize_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detector.process_frame(frame)

        if CameraUtils.display_frame("Hand Landmark Detection", processed_frame) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
