import time
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from face_roi import FaceROIExtractor
from pos_algorithm import POSAlgorithm


class HeartRateMonitor:

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.roi_extractor = FaceROIExtractor()

        buf_len = int(Config.BUFFER_DURATION * Config.FPS_TARGET)
        self.r_buffer = deque(maxlen=buf_len)
        self.g_buffer = deque(maxlen=buf_len)
        self.b_buffer = deque(maxlen=buf_len)
        self.timestamps = deque(maxlen=buf_len)

        self.window_size = int(Config.WINDOW_DURATION * Config.FPS_TARGET)
        self.frame_count = 0
        self.current_hr = 0.0
        self.hr_history = deque(maxlen=30)
        self.face_detected = False

        self.pos = None
        self.fig = None
        self.axes = None
        self._setup_plots()

    def _setup_plots(self):
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('Real-Time rPPG Heart Rate Monitor — POS Algorithm', fontsize=13)
        plt.ion()
        self.fig.show()

    def _compute_hr(self):
        if len(self.r_buffer) < self.window_size:
            return 0.0

        r = list(self.r_buffer)[-self.window_size:]
        g = list(self.g_buffer)[-self.window_size:]
        b = list(self.b_buffer)[-self.window_size:]

        ts = list(self.timestamps)[-self.window_size:]
        actual_fps = (
            len(ts) / (ts[-1] - ts[0])
            if (ts[-1] - ts[0]) > 1e-6
            else Config.FPS_TARGET
        )

        self.pos = POSAlgorithm(
            fs=actual_fps,
            lowcut=Config.LOW_HR_HZ,
            highcut=Config.HIGH_HR_HZ,
            order=Config.FILTER_ORDER,
        )

        hr, _ = self.pos.get_heart_rate(r, g, b)
        return hr

    def _update_plots(self):
        if self.fig is None:
            self._setup_plots()

        ax1, ax2, ax3 = self.axes
        for ax in self.axes:
            ax.clear()

        if len(self.r_buffer) > 10:
            ts = list(self.timestamps)
            ax1.plot(ts, list(self.r_buffer), 'r', alpha=0.7, lw=0.8, label='R')
            ax1.plot(ts, list(self.g_buffer), 'g', alpha=0.7, lw=0.8, label='G')
            ax1.plot(ts, list(self.b_buffer), 'b', alpha=0.7, lw=0.8, label='B')
            ax1.set_ylabel('Mean RGB')
            ax1.legend(loc='upper right')
            ax1.set_title('Raw RGB Signals from ROI')

        if len(self.r_buffer) >= self.window_size:
            r = list(self.r_buffer)[-self.window_size:]
            g = list(self.g_buffer)[-self.window_size:]
            b = list(self.b_buffer)[-self.window_size:]

            if self.pos is None:
                self.pos = POSAlgorithm(fs=Config.FPS_TARGET)
            pulse = self.pos.process(r, g, b)
            ax2.plot(pulse, 'purple', lw=1.0)
            ax2.set_ylabel('Amplitude')
            ax2.set_xlabel('Frame')
            ax2.set_title('Extracted Pulse Signal (POS Algorithm)')

        if len(self.hr_history) > 0:
            hrs = list(self.hr_history)
            ax3.plot(hrs, 'green', lw=1.0)
            ax3.set_ylabel('Heart Rate (BPM)')
            ax3.set_xlabel('Measurement #')
            ax3.set_ylim([40, 200])
            mean_hr = np.mean(hrs)
            ax3.axhline(y=mean_hr, color='red', ls='--', alpha=0.5)
            ax3.set_title(
                f'Current HR: {self.current_hr:.1f} BPM  |  Avg: {mean_hr:.1f} BPM'
            )

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_info(self, frame):
        if self.face_detected:
            if self.current_hr > 0:
                msg = f'HR: {self.current_hr:.1f} BPM'
                color = (0, 255, 0)
            elif len(self.r_buffer) < self.window_size:
                pct = len(self.r_buffer) / self.window_size * 100
                msg = f'Buffering: {pct:.0f}%'
                color = (255, 255, 0)
            else:
                msg = 'Calculating...'
                color = (0, 255, 255)

            cv2.putText(frame, msg, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            if len(self.hr_history) > 0:
                avg = np.mean(self.hr_history)
                cv2.putText(frame, f'Avg: {avg:.1f} BPM', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(frame, 'No Face Detected', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f'Error: Could not open camera {self.camera_id}')
            return

        print('=' * 55)
        print('  Heart Rate Monitor — rPPG with POS Algorithm')
        print('=' * 55)
        print('  Controls:')
        print('    q  – Quit')
        print('    r  – Reset buffers')
        print('    p  – Toggle plots (on/off)')
        print('=' * 55)

        start = time.time()
        show_plots = True

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = time.time() - start
            self.frame_count += 1

            rgb_vals, mask = self.roi_extractor.extract_rgb(frame)

            if rgb_vals is not None and mask is not None:
                self.face_detected = True
                r, g, b = rgb_vals
                self.r_buffer.append(r)
                self.g_buffer.append(g)
                self.b_buffer.append(b)
                self.timestamps.append(t)
                frame = self.roi_extractor.visualize_roi(frame, mask)
            else:
                self.face_detected = False

            if (
                self.face_detected
                and len(self.r_buffer) >= self.window_size
                and self.frame_count % Config.HR_UPDATE_INTERVAL == 0
            ):
                hr = self._compute_hr()
                if hr > 0:
                    self.current_hr = hr
                    self.hr_history.append(hr)

            self._draw_info(frame)
            cv2.imshow('rPPG Heart Rate Monitor', frame)

            if show_plots and self.frame_count % Config.PLOT_UPDATE_INTERVAL == 0:
                self._update_plots()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.r_buffer.clear()
                self.g_buffer.clear()
                self.b_buffer.clear()
                self.timestamps.clear()
                self.hr_history.clear()
                self.current_hr = 0.0
                print('Buffers reset.')
            elif key == ord('p'):
                show_plots = not show_plots
                if not show_plots:
                    plt.ioff()
                    plt.close('all')
                    self.fig = None
                    self.axes = None

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close('all')
        print('\nMonitoring stopped.')
