from imutils import contours
import numpy as np
import argparse
import cv2
import myutils


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


template = cv2.imread('images/template.png')
cv_show('template', template)

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cv_show('templage_gray', template_gray)

template_binary = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('template_binary', template_binary)

template_fCnts, cnts, hierarchy = cv2.findContours(template_binary
                                                   , cv2.RETR_EXTERNAL
                                                   , cv2.CHAIN_APPROX_SIMPLE)
template_rect = cv2.drawContours(template.copy(), cnts, -1, (0, 0, 255), 2)
cv_show('template_rect', template_rect)
cnts = myutils.sort_contours(cnts, method="left-to-right")[0]
number = {}

for (i, cnt) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(cnt)
    roi = template_binary[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    number[i] = roi

cardImg = cv2.imread('images/credit_card_01.png')
cardImg = cv2.resize(cardImg
                     , (300, int(float(300 / cardImg.shape[1]) * cardImg.shape[0]))
                     , interpolation=cv2.INTER_AREA)
cv_show('cardImg', cardImg)
cardImg_gray = cv2.cvtColor(cardImg, cv2.COLOR_BGR2GRAY)
cv_show('cardImg_gray', cardImg_gray)
# cardImg_binary = cv2.threshold(cardImg_gray, 0, 255, cv2.THRESH_BINARY)[1]
# cv_show('cardImg_binary', cardImg_binary)

# 指定卷积核大小
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

cardImg_tophat = cv2.morphologyEx(cardImg_gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('cardImg_open', cardImg_tophat)

sobelx = cv2.Sobel(cardImg_tophat, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
(minX, maxX) = (np.min(sobelx), np.max(sobelx))
sobelx = (255 * ((sobelx - minX) / (maxX - minX)))
sobelx = sobelx.astype('uint8')

# sobely = cv2.Sobel(cardImg_tophat, cv2.CV_64F, 0, 1, ksize=3)
# sobely = cv2.convertScaleAbs(sobely)
# (minY, maxY) = (np.min(sobely), np.max(sobely))
# sobely = (255 * ((sobely - minY) / (maxY - minY)))
# # sobely = sobely.astype('uint8')
#
# sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# sobelxy = sobelxy.astype('uint8')
cv_show('sobelx', sobelx)

cardImg_close = cv2.morphologyEx(sobelx, cv2.MORPH_CLOSE, rectKernel)
cv_show('cardImg_close', cardImg_close)

cardImg_binary = cv2.threshold(cardImg_close, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
cv_show('cardImg_binary', cardImg_binary)

cardImg_close = cv2.morphologyEx(cardImg_binary, cv2.MORPH_CLOSE, sqKernel)
cv_show('cardImg_close', cardImg_close)

cardImg_, cnts, hierarchy = cv2.findContours(cardImg_close
                                             , cv2.RETR_EXTERNAL
                                             , cv2.CHAIN_APPROX_SIMPLE)
cardImg_cnts = cv2.drawContours(cardImg.copy(), cnts, -1, (0, 0, 255), 2)
cv_show('cardImg_cnts', cardImg_cnts)

# 对轮廓进行筛选
locs = []
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])

output = []
for (i, (x, y, w, h)) in enumerate(locs):
    group_output = []
    group = cardImg_gray[y-5:y + h + 5, x-5:x + w + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)[1]
    cv_show('group', group)
    group_, group_cnts, group_hierarchy = cv2.findContours(group
                                                           , cv2.RETR_EXTERNAL
                                                           , cv2.CHAIN_APPROX_SIMPLE)
    group_cnts = contours.sort_contours(group_cnts, method="left-to-right")[0]
    for cnt in group_cnts:
        (nx,ny,nw,nh) = cv2.boundingRect(cnt)
        roi = group[ny:ny+nh, nx:nx+nw]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        score = []

        for (number_i, number_roi) in number.items():
            result = cv2.matchTemplate(roi, number_roi, cv2.TM_CCOEFF)
            score_ = cv2.minMaxLoc(result)[1]

            score.append(score_)


        group_output.append(str(np.argmax(score)))

    cv2.rectangle(cardImg, (x - 5, y - 5),
                  (x + w + 5, y + h + 5), (0, 0, 255), 1)
    cv2.putText(cardImg, "".join(group_output), (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.append(group_output)

cv_show('cardImg', cardImg)