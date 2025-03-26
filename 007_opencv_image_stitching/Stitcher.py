import cv2
import numpy as np
import utils

def _detect_describe(image):
    # SIFT特征检测器
    descriptor = cv2.SIFT_create()

    # 寻找图像中的特征点，计算这些关键点的特征描述子，用于后续的匹配
    (kps, features) = descriptor.detectAndCompute(image, None)

    # 关键点的坐标
    num_kps = np.float32([kp.pt for kp in kps])

    # 关键点的坐标，关键点的描述子，关键点对象
    return num_kps, features, kps

def _match_key_points(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # 暴力匹配器
    matcher = cv2.BFMatcher()

    # k-近邻匹配，2 每个特征点，在另一张图中找到两个最接近的匹配点
    # 计算两者的欧几里得距离，如果最近的匹配点比次近的匹配点明显更近（由 ratio 参数控制），则认为匹配成功
    raw_matches = matcher.knnMatch(featuresA, featuresB, 2)

    matches = []
    for m in raw_matches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # 当筛选后的匹配对大于4时，计算视角变换矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        pts_a = np.float32([kpsA[i] for (_, i) in matches])
        pts_b = np.float32([kpsB[i] for (i, _) in matches])

        # 通过RANSAC筛选出更准确的匹配点对，减少误匹配的影响
        (H, status) = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reprojThresh)

        # 返回结果
        return matches, H, status

    # 如果匹配对小于4时，返回None
    return None

def _draw_matches(imageA, imageB, kpsA, kpsB, matches, status):
    # 这里将两张图片横向拼接，并在匹配点之间绘制红色直线，帮助可视化匹配效果

    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            pt_a = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
            pt_b = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
            cv2.line(vis, pt_a, pt_b, (0, 0, 255), 1)
    return vis

# 拼接函数
def stitch(imageB, imageA, ratio=0.75, reprojThresh=4.0):
    # 提取SIFT特征：对两张图像分别检测特征点和计算描述子
    (num_kpsA, featuresA, kpsA) = _detect_describe(imageA)
    (num_kpsB, featuresB, kpsB) = _detect_describe(imageB)

    # 找到匹配点对，并计算 单应性矩阵 H
    m = _match_key_points(num_kpsA, num_kpsB, featuresA, featuresB, ratio, reprojThresh)

    # 如果返回结果为空，没有匹配成功的特征点，退出算法
    if m is None:
        return None

    # H是3x3视角变换矩阵
    (matches, H, status) = m
    # 透视变换
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    utils.cv_show('result imageA', result)
    # 将图片B传入result图片最左端
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    utils.cv_show('result imageB', result)
    # 生成匹配图片
    vis = _draw_matches(imageA, imageB, kpsA, kpsB, matches, status)
    utils.cv_show('draw matches', vis)
    return result
