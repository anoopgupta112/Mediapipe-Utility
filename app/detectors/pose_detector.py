import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.drawing_utils = mp.solutions.drawing_utils

        self.green_spec = self.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.green_spec,  # Green landmarks
                connection_drawing_spec=self.green_spec  # Green connections
            )
        return frame
