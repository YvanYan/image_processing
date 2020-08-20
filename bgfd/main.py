import cv2

cap = cv2.VideoCapture('test.avi')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while (True):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 188:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    k = cv2.waitKey(150)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
