import cv2
import mediapipe as mp


class HandLandmarkDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.green_color = self.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    def process_frame(self, frame):
        """Process frame and draw hand landmarks in green."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.green_color,
                    self.green_color
                )
        return frame
