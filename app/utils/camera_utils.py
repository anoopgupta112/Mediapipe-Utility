import cv2

class CameraUtils:
    @staticmethod
    def initialize_camera(camera_id=0):
        """Initialize camera capture."""
        return cv2.VideoCapture(camera_id)

    @staticmethod
    def process_frame(frame):
        """Convert BGR frame to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def display_frame(window_name, frame):
        """Display frame in a window."""
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1)
