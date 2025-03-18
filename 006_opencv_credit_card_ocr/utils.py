import cv2

def sort_contours(contours, method="left-to-right"):
    """
    对轮廓进行排序，确保后续的识别顺序正确（如从左到右或从上到下）
    :param contours:
    :param method:
    :return:
    """
    # 1️⃣ 确定排序方式，默认升序，从左到右从上到下
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 2️⃣ 计算轮廓的外接矩形，bounding_boxes轮廓位置大小
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # 3️⃣ 按指定方向排序，将轮廓和对应的边界框绑定在一起
    (_contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))
    return _contours, bounding_boxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    按比例缩放图片，支持指定宽度或高度 进行等比例缩放，避免图片失真
    :param image: 输入图像（numpy 数组）
    :param width: 目标 宽度，如果设为 None，则按 高度等比例缩放
    :param height: 目标 高度，如果设为 None，则按 宽度等比例缩放
    :param inter: 插值方法，默认使用 cv2.INTER_AREA（适用于缩小图片）
    :return: 图像（numpy 数组）
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
