import numpy as np

try:
    from scipy import signal
    from scipy.signal import butter, filtfilt
except Exception:
    signal = None
    butter = None
    filtfilt = None


class SignalProcessor:

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=4):
        if butter is None:
            return None, None

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        data = np.asarray(data, dtype=np.float64)

        if filtfilt is not None:
            b, a = SignalProcessor.butter_bandpass(lowcut, highcut, fs, order)
            return filtfilt(b, a, data)

        if data.size < 4:
            return data

        freqs = np.fft.rfftfreq(data.size, d=1 / fs)
        spectrum = np.fft.rfft(data)
        mask = (freqs >= lowcut) & (freqs <= highcut)
        return np.fft.irfft(spectrum * mask, n=data.size)

    @staticmethod
    def detrend(data):
        data = np.asarray(data, dtype=np.float64)

        if signal is not None:
            return signal.detrend(data)

        if data.size < 2:
            return data - np.mean(data)

        x = np.arange(data.size, dtype=np.float64)
        slope, intercept = np.polyfit(x, data, 1)
        return data - (slope * x + intercept)

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
