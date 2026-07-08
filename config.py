class Config:
    # Length of the signal window used to estimate heart rate. The POS algorithm and FFT operate on the last 6 seconds of data.
    WINDOW_DURATION = 6.0 
    # Length of the signal window used to estimate heart rate. The POS algorithm and FFT operate on the last 6 seconds of data.
    BUFFER_DURATION = 10.0
    # Expected camera frame rate (30 frames per second). Used for timing and signal processing.
    FPS_TARGET = 30

    # Lower cutoff frequency of the band-pass filter. Removes frequencies below about 42 BPM (0.7 × 60).
    LOW_HR_HZ = 0.7
    # Upper cutoff frequency of the band-pass filter. Removes frequencies above about 180 BPM (3.0 × 60).
    HIGH_HR_HZ = 3.0


    # Order of the Butterworth band-pass filter. A higher order provides sharper filtering but increases computational complexity.
    FILTER_ORDER = 4

    # Shrinks the facial mask by 2% to remove boundary pixels, reducing contamination from hair, background, and facial edges.
    ROI_EROSION_RATIO = 0.02

    # Heart rate is recalculated every 5 frames instead of every frame to reduce computation and stabilize results.
    HR_UPDATE_INTERVAL = 5
    # Signal plots are refreshed every 10 frames to improve UI performance.
    PLOT_UPDATE_INTERVAL = 10
