import argparse
from mi4l.io.video import get_video_info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a video file")
    args = ap.parse_args()

    info = get_video_info(args.video)
    print(info)

if __name__ == "__main__":
    main()