from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time
import dlib
import cv2

# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

facial_landmarks = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def _shape_to_np(shape, dtype="int"):
    # 创建68*2
    coors = np.zeros((shape.num_parts, 2), dtype=dtype)
    # 遍历每一个关键点
    # 得到坐标
    for i in range(0, shape.num_parts):
        coors[i] = (shape.part(i).x, shape.part(i).y)
    return coors

def _eye_aspect_ratio(eye):
    # 计算距离，竖直的
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算距离，水平的
    C = dist.euclidean(eye[0], eye[3])
    # ear值
    ear = (A + B) / (2.0 * C)
    return ear


def detect_blinks(path):
    eye_ar_thresh = 0.3
    eye_ar_consec_frames = 3

    counter = 0
    total = 0

    # 检测与定位工具
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 分别取两个眼睛区域
    (lStart, lEnd) = facial_landmarks["left_eye"]
    (rStart, rEnd) = facial_landmarks["right_eye"]

    # 读取视频
    print("[INFO] starting video stream thread...")
    vs = cv2.VideoCapture(path)
    time.sleep(1.0)

    # 遍历每一帧
    while True:
        # 预处理
        frame = vs.read()[1]
        if frame is None:
            break

        (h, w) = frame.shape[:2]
        width = 1200
        r = width / float(w)
        dim = (width, int(h * r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        rects = detector(gray, 0)

        # 遍历每一个检测到的人脸
        for rect in rects:
            # 获取坐标
            shape = predictor(gray, rect)
            shape = _shape_to_np(shape)

            # 分别计算ear值
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = _eye_aspect_ratio(leftEye)
            rightEAR = _eye_aspect_ratio(rightEye)

            # 算一个平均的
            ear = (leftEAR + rightEAR) / 2.0

            # 绘制眼睛区域
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # 检查是否满足阈值
            if ear < eye_ar_thresh:
                counter += 1

            else:
                # 如果连续几帧都是闭眼的，总数算一次
                if counter >= eye_ar_consec_frames:
                    total += 1

                # 重置
                counter = 0

            # 显示
            cv2.putText(frame, "Blinks: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:
            break

    vs.release()
    cv2.destroyAllWindows()
