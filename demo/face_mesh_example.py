from app.detectors.face_mesh_detector import FaceMeshDetector
from app.utils.camera_utils import CameraUtils
import cv2
def main():
    detector = FaceMeshDetector()
    cap = CameraUtils.initialize_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detector.process_frame(frame)

        if CameraUtils.display_frame("Face Mesh Detection", processed_frame) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
