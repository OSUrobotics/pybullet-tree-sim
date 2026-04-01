#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import time
import queue


class RGBStreamVisualizer:
    def __init__(self, window_name: str = "RGBStream", max_queue_size: int = 2, use_thread: bool = False) -> None:

        self.window_name = window_name
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        self.use_thread = use_thread
        return

    def start(self) -> None:
        if self.running:
            return

        self.running = True
        # create the window once on start and run display loop in background
        cv2.namedWindow(winname=self.window_name, flags=cv2.WINDOW_NORMAL)
        if self.use_thread:
            # self.thread = threading.Thread(target=self.display_loop, daemon=True)
            # self.thread.start()
            ...
        return

    def stop(self) -> None:
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
            self.thread = None
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        return

    def update_frame(self, rgb_data: np.ndarray) -> None:
        # Convert float arrays to uint8 if needed
        if rgb_data is None:
            return

        if rgb_data.dtype == np.float32 or rgb_data.dtype == np.float64:
            rgb_data = np.clip(rgb_data, 0.0, 1.0)
            rgb_data = (rgb_data * 255).astype(np.uint8)
        elif rgb_data.dtype != np.uint8:
            try:
                rgb_data = rgb_data.astype(np.uint8)
            except Exception:
                return

        # drop alpha if present
        if rgb_data.ndim == 3 and rgb_data.shape[2] == 4:
            rgb_data = rgb_data[..., :3]

        # OpenCV uses BGR
        try:
            bgr_data = cv2.cvtColor(src=rgb_data, code=cv2.COLOR_RGB2BGR)
        except Exception:
            bgr_data = rgb_data

        try:
            self.frame_queue.put_nowait(item=bgr_data)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(item=bgr_data)
            except queue.Empty:
                pass

        return

    def display_loop(self) -> None:
        """Internal method: runs in separate thread to display frames."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                try:
                    cv2.imshow(winname=self.window_name, mat=frame)
                except Exception:
                    pass

                key = cv2.waitKey(delay=1)
                if key == 27:  # esc key
                    self.running = False
                    break
            except queue.Empty:
                # no frame available, just process events
                cv2.waitKey(delay=1)

        return
