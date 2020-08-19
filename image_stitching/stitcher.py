import numpy as np
import cv2


class stitcher:

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (imageA, imageB) = images

        # 找特征点
        (kpsA, npkpsA, feasA) = self.find_kps_feas(imageA)
        (kpsB, npkpsB, feasB) = self.find_kps_feas(imageB)

        M = self.matchKeypoints(npkpsA, npkpsB, feasA, feasB, ratio, reprojThresh)
        (good, H, status) = M
        result = cv2.warpPerspective(imageB, H, (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
        self.cv_show('result1', result)
        result[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
        self.cv_show('result2', result)

        img = cv2.drawMatchesKnn(imageB, kpsB, imageA, kpsA, good, None, flags=2)
        self.cv_show('result', img)

        return result, img

    def find_kps_feas(self, image):
        # 建立SIFT生成器
        descriptor = cv2.xfeatures2d.SIFT_create()
        # 检测SIFT特征点
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将结果转换成NumPy数组
        npkps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return (kps, npkps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        bf = cv2.BFMatcher()
        allMatches = bf.knnMatch(featuresB, featuresA, k=2)
        matches = []
        good = []
        for m, n in allMatches:
            if m.distance < ratio * n.distance:
                matches.append((m.trainIdx, m.queryIdx))
                good.append([m])

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[i] for (_, i) in matches])

            (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)

        return (good, H, status)

    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
