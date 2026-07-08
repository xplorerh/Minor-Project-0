import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt


class SignalProcessor:

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = SignalProcessor.butter_bandpass(lowcut, highcut, fs, order)
        return filtfilt(b, a, data)

    @staticmethod
    def detrend(data):
        return signal.detrend(data)

    @staticmethod
    def compute_hr_from_fft(signal_data, fs, low_hr=0.7, high_hr=3.0):
        n = len(signal_data)
        if n < 10:
            return 0.0

        windowed = signal_data * np.hanning(n)

        freqs = np.fft.rfftfreq(n, d=1 / fs)
        fft_vals = np.fft.rfft(windowed)

        mask = (freqs >= low_hr) & (freqs <= high_hr)
        freqs_hr = freqs[mask]
        fft_hr = fft_vals[mask]

        if len(fft_hr) == 0:
            return 0.0

        idx = np.argmax(np.abs(fft_hr))
        return freqs_hr[idx] * 60.0
