import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):
        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.drawing_utils.draw_landmarks(
                    frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_styles.get_default_face_mesh_tesselation_style()
                )

                self.drawing_utils.draw_landmarks(
                    frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_styles.get_default_face_mesh_contours_style()
                )

        return frame
