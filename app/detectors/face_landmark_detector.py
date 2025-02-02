import cv2
import mediapipe as mp


class FaceLandmarkDetector:
    def __init__(self, static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence
        )

    def process_frame(self, frame):
        height, width, _ = frame.shape
        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            for facial_landmarks in results.multi_face_landmarks:
                for i in range(0, 468):
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
        return frame

