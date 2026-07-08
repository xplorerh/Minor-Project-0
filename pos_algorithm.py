import numpy as np
from signal_processing import SignalProcessor


class POSAlgorithm:
    """
    Plane Orthogonal to Skin (POS) rPPG Algorithm

    Based on: Wang et al., "Algorithmic Principles of Remote PPG" (2017)

    Projects normalized RGB signals onto a plane orthogonal to the
    skin-tone vector (1,1,1), then extracts the pulse signal from
    the projected components via adaptive combination and bandpass
    filtering.
    """

    def __init__(self, fs=30, lowcut=0.7, highcut=3.0, order=4):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.sig_proc = SignalProcessor()

    def process(self, r, g, b):
        r = np.array(r, dtype=np.float64)
        g = np.array(g, dtype=np.float64)
        b = np.array(b, dtype=np.float64)

        r = self.sig_proc.detrend(r)
        g = self.sig_proc.detrend(g)
        b = self.sig_proc.detrend(b)

        rn = r / (np.mean(r) + 1e-10)
        gn = g / (np.mean(g) + 1e-10)
        bn = b / (np.mean(b) + 1e-10)

        s1 = 0.5 * rn + 0.5 * gn - bn
        s2 = -rn + gn

        s1 = s1 - np.mean(s1)
        s2 = s2 - np.mean(s2)

        alpha = np.std(s1) / (np.std(s2) + 1e-10)
        pulse = s1 - alpha * s2

        pulse = self.sig_proc.bandpass_filter(
            pulse, self.lowcut, self.highcut, self.fs, self.order
        )

        return pulse

    def get_heart_rate(self, r, g, b):
        pulse = self.process(r, g, b)
        hr = self.sig_proc.compute_hr_from_fft(
            pulse, self.fs, self.lowcut, self.highcut
        )
        return hr, pulse
