from app.detectors.face_landmark_detector import FaceLandmarkDetector
from app.utils.camera_utils import CameraUtils
import cv2


def main():
    detector = FaceLandmarkDetector()
    cap = CameraUtils.initialize_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detector.process_frame(frame)

        if CameraUtils.display_frame("Face Landmark Detection", processed_frame) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
