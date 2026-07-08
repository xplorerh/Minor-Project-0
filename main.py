import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from heartrate_monitor import HeartRateMonitor


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Heart Rate Monitoring via rPPG (POS Algorithm)'
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--window', type=float, default=6.0,
                        help='Analysis window in seconds (default: 6.0)')
    parser.add_argument('--low-hr', type=float, default=0.7,
                        help='Low HR cutoff in Hz, default 0.7 (~42 BPM)')
    parser.add_argument('--high-hr', type=float, default=3.0,
                        help='High HR cutoff in Hz, default 3.0 (~180 BPM)')
    parser.add_argument('--erosion', type=float, default=0.02,
                        help='ROI erosion ratio (default: 0.02)')

    args = parser.parse_args()

    Config.WINDOW_DURATION = args.window
    Config.LOW_HR_HZ = args.low_hr
    Config.HIGH_HR_HZ = args.high_hr
    Config.ROI_EROSION_RATIO = args.erosion

    monitor = HeartRateMonitor(camera_id=args.camera)

    try:
        monitor.run()
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
