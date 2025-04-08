import cv2

def get_tracker(name):
    """获取兼容不同OpenCV版本的跟踪器构造函数"""
    if hasattr(cv2.legacy, name + "_create"):
        return getattr(cv2.legacy, name + "_create")
    elif hasattr(cv2, name + "_create"):
        return getattr(cv2, name + "_create")
    return None


# opencv已经实现了的追踪算法
opencv_object_trackers = {
    "boosting": get_tracker("TrackerBoosting"),  # 基于AdaBoost的传统算法，性能较差
    "mil": get_tracker("TrackerMIL"),  # 比Boosting鲁棒，但仍有局限性
    "kcf": get_tracker("TrackerKCF"),  # 速度快，但对遮挡敏感
    "tld": get_tracker("TrackerTLD"),  # 处理目标遮挡和消失，但易漂移
    "medianflow": get_tracker("TrackerMedianFlow"),  # 适用于匀速运动的小目标，失败时会自我检测
    "mosse": get_tracker("TrackerMOSSE"),  # 极快，适合实时应用，但精度较低
    "csrt": get_tracker("TrackerCSRT")  # 高精度，但速度较慢
}

def tracking(video_path, tracker_type="kcf"):
    # 实例化 multi-object tracker
    trackers = cv2.legacy.MultiTracker_create()
    vs = cv2.VideoCapture(video_path)

    # 视频流
    while True:
        # 取当前帧
        frame = vs.read()
        frame = frame[1]
        if frame is None:
            break

        # resize每一帧
        (h, w) = frame.shape[:2]
        width = 600
        r = width / float(w)
        dim = (width, int(h * r))
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # 追踪结果
        (success, boxes) = trackers.update(frame)

        # 绘制区域
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(100) & 0xFF

        if key == ord("s"):
            # 选择一个区域，按s
            box = cv2.selectROI("Frame", frame, fromCenter=False,
                                showCrosshair=True)

            # 创建一个新的追踪器
            tracker = opencv_object_trackers[tracker_type]()
            trackers.add(tracker, frame, box)

        # 退出
        elif key == 27:
            break
    vs.release()
    cv2.destroyAllWindows()
