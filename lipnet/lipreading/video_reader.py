import cv2
import numpy as np

class VideoReader:
    """A wrapper class for video reading that uses OpenCV internally."""
    
    @staticmethod
    def read_video(path):
        """Read a video file and return frames as a numpy array.
        
        Args:
            path (str): Path to the video file
            
        Returns:
            numpy.ndarray: Array of frames in shape (T, H, W, C)
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        return np.array(frames)
    
    @staticmethod
    def vreader(path):
        """Generator function that yields frames from a video file.
        
        Args:
            path (str): Path to the video file
            
        Yields:
            numpy.ndarray: Individual frames in shape (H, W, C)
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
            
        cap.release() 