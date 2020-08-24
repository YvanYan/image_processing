import cv2

# opencv已经实现了的追踪算法
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

cap = cv2.VideoCapture('videos/soccer_01.mp4')
trackers = cv2.MultiTracker_create()

while (True):
    frame = cap.read()[1]  # get frame, the first return of read() is a bool variable , the second is frame
    if frame is None:
        break

    (h, w) = frame.shape[0:2]
    width = 600
    ratio = width / float(w)
    height = int(h * ratio)
    frame = cv2.resize(frame, (height, width), interpolation=cv2.INTER_AREA)

    (success, boxs) = trackers.update(frame)

    for box in boxs:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(100)

    if key == ord('s'):
        box = cv2.selectROI("frame", frame, fromCenter=False,
                            showCrosshair=True)
        tracker = OPENCV_OBJECT_TRACKERS['kcf']()
        trackers.add(tracker, frame, box)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
