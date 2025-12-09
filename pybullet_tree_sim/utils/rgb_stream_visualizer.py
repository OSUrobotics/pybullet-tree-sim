#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import time
import queue


class RGBStreamVisualizer:
    def __init__(self, window_name: str = "RGBStream", max_queue_size: int = 2) -> None:

        self.window_name = window_name
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None

        return

    def start(self) -> None:
        if self.running:
            return

        self.running = True
        #     self.thread = threading.Thread(target=self._display_loop, daemon=True)
        #     self.thread.start()
        return

    def stop(self) -> None:
        self.running = False
        #     if self.thread:
        #         self.thread.join()
        cv2.destroyWindow(self.window_name)
        return

    def update_frame(self, rgb_data: np.ndarray) -> None:
        # Convert float arrays to uint8 if needed
        if rgb_data.dtype == np.float32 or rgb_data.dtype == np.float64:
            rgb_data = (rgb_data * 255).astype(np.uint8)

        # OpenCV uses BGR, so convert
        bgr_data = cv2.cvtColor(src=rgb_data, code=cv2.COLOR_RGB2BGR)

        try:
            self.frame_queue.put_nowait(item=bgr_data)
        except queue.Full:
            # Remove oldest, add new
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(item=bgr_data)
            except queue.Empty:
                # print("FAIL")
                pass

        self._display_loop()
        return

    def _display_loop(self) -> None:
        """Internal method: runs in separate thread to display frames."""
        cv2.namedWindow(winname=self.window_name, flags=cv2.WINDOW_NORMAL)
        # while self.running:
        try:
            # Get frame with timeout
            frame = self.frame_queue.get(timeout=0.1)
            print(frame)
            cv2.imshow(winname=self.window_name, mat=frame)

            # Process window events
            key = cv2.waitKey(delay=1)
            if key == 27:  # esc key
                self.running = False

        except queue.Empty:
            # no frame available, just process events
            cv2.waitKey(delay=1)

        return
