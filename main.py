from CascadeDetector import CascadeDetector as CD
from DrawResult import draw_rects
import cv2
import os

opencv_cascade_filepath = "./cascades/opencv_logo_cascade.xml"
unn_cascade_filepath = "./cascades/unn_old_logo_cascade.xml"
intel_cascade_filepath = "./cascade/intel_logo_cascade.xml"

video_filepath = "./video/logo.mp4"

if __name__ == "__main__":
    opencvCD = CD(opencv_cascade_filepath)
    unnCD = CD(unn_cascade_filepath)
    intelCD = CD(intel_cascade_filepath)

    winname = os.path.basename(video_filepath)

    while True:
        video_cap = cv2.VideoCapture(video_filepath)
        ret, frame = video_cap.read()
        if not ret:
            break
        rects = opencvCD.detect(frame)
        # print(rects)
        img_to_show = frame
        img_to_show = draw_rects(img_to_show, opencvCD.detect(frame))
        img_to_show = draw_rects(img_to_show, unnCD.detect(frame))
        # img_to_show = draw_rects(img_to_show, intelCD.detect(frame))

        cv2.imshow(winname, img_to_show)
        cv2.waitKey(0)
        # break
