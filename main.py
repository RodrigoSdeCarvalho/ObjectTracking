import cv2
import os
from cv2 import VideoCapture
import sys

ASSETS_PATH = os.path.join(os.getcwd(), "Assets")

def main(argv) -> None:
    """Main function of the program. Runs object tracking on the video file"""    
    video = cv2.VideoCapture(os.path.join(os.getcwd(), "Assets", argv[1]))
    ok, frame = video.read()
    tracker = cv2.TrackerCSRT_create()
    selected_object = cv2.selectROI(frame)

    track_object(video, tracker, selected_object, frame)


def track_object(video:VideoCapture, tracker, selected_object, frame) -> None:
    """Tracks the selected object in the video file

    Args:
        video (VideoCapture): Video file to track the object in
        tracker (_type_): Model used to track the object
        selected_object (_type_): Object to track
    """
    ok = tracker.init(frame, selected_object)

    while True:
        ok, frame = video.read()

        if not ok:
            break

        ok, selected_object = tracker.update(frame)
        if ok:
            (x, y, w, h) = [int(v) for v in selected_object]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2, 1)
        else:
            cv2.putText(frame, 'Error', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0XFF == 27: # ESC
            break


if __name__ == '__main__':
    main(sys.argv)
