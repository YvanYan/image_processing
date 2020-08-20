import cv2
import numpy as np

cap = cv2.VideoCapture('test.avi')
# 角点检测参数
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)

# lucas kanade参数
lk_params = dict(winSize=(15, 15), maxLevel=2)

color = np.random.randint(0, 255, (100, 3))

ret, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)

mask = np.zeros_like(first_frame)

while (True):
    ret, cur_frame = cap.read()
    cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray, cur_frame_gray, p0, None, **lk_params)

    good_cur = p1[st == 1]
    good_first = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_cur, good_first)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(cur_frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(cur_frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(150) & 0xff
    if k == 27:
        break

    # 更新
    first_gray = cur_frame_gray.copy()
    # p0 = good_cur.reshape(-1,1,2)
    p2 = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)
    if abs(len(p2) - len(p0)) >= 5:
        p0 = p2
    else:
        p0 = p1

cv2.destroyAllWindows()
cap.release()
