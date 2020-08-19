import cv2
import numpy as np

# 正确答案
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_pts(pts):
    rect = np.zeros((4, 2), dtype='float32')

    # 左上，右上，右下，左下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_pts_transform(image, pts):
    rect = get_pts(pts)
    (tl, tr, br, bl) = rect

    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bot = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthMax = max(int(width_bot), int(width_top))

    height_left = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    height_right = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    heightMax = max(int(height_left), int(height_right))

    dest = np.array([[0, 0], [widthMax - 1, 0], [widthMax - 1, heightMax - 1], [0, heightMax - 1]], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dest)
    warped = cv2.warpPerspective(image, M, (widthMax, heightMax))
    return warped


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes


image = cv2.imread('images\\test_02.png')
img_copy = image.copy()
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
cv_show('img_blur', img_blur)
edge = cv2.Canny(img_blur, 75, 200)
cv_show('edge', edge)

cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(img_copy, cnts, -1, (0, 0, 255), 3)
cv_show('drawCnts', img_copy)
cnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            cnt = approx
            break

warped = get_pts_transform(img_gray, cnt.reshape(4, 2))
cv_show('warped', warped)
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)
thresh_copy = thresh.copy()
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
cv2.drawContours(thresh_copy, cnts, -1, (0, 0, 255), 3)
cv_show('thresh_drawCnts', thresh_copy)
recordCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = w / float(h)
    if w >= 20 and h >= 20 and ratio >= 0.9 and ratio <= 1.1:
        recordCnts.append(c)

recordCnts = sort_contours(recordCnts, method="top-to-bottom")[0]
result = 0

for (q, i) in enumerate(np.arange(0, len(recordCnts), 5)):
    cnts = sort_contours(recordCnts[i:i + 5])[0]
    bubbled = None

    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)
        cv_show('mask', mask)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    k = ANSWER_KEY[q]
    if k == bubbled[1]:
        cv2.drawContours(warped, [cnts[k]], -1, (0, 255, 0), 3)
        result +=1
    else:
        cv2.drawContours(warped, [cnts[k]], -1, (0, 0, 255), 3)

score = (result / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(warped, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)