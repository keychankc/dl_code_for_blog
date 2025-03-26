import cv2
import numpy as np
import random
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_rgb_white_yellow(image):
    # 提取出白色和黄色的区域，并返回一个掩码后的图像

    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    # 提取lower和upper之间颜色
    # mask是一个二值图像，白色区域表示符合颜色范围的像素，黑色区域表示不符合的像素
    mask = cv2.inRange(image, lower, upper)
    # 保留掩码中白色区域对应的像素，其余区域置为黑色
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

def convert_gray(image):
    # 转灰度图
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def detect_edges(image, low_threshold=50, high_threshold=200):
    # 边缘检测 low_threshold：低阈值，用于边缘连接  high_threshold：高阈值，用于强边缘检测
    # 返回一个二值图像，其中边缘像素为白色（255），非边缘像素为黑色（0）
    return cv2.Canny(image, low_threshold, high_threshold)


def select_region(image):
    rows, cols = image.shape[:2]
    pt_1 = [int(cols * 0.05), int(rows * 0.90)]
    pt_2 = [int(cols * 0.05), int(rows * 0.70)]
    pt_3 = [int(cols * 0.30), int(rows * 0.55)]
    pt_4 = [int(cols * 0.6), int(rows * 0.13)]
    pt_5 = [int(cols * 0.90), int(rows * 0.15)]
    pt_6 = [int(cols * 0.90), int(rows * 0.95)]

    vertices = np.array([pt_1, pt_2, pt_3, pt_4, pt_5, pt_6], dtype=np.int32)
    cv_vertices = [vertices.reshape(-1, 1, 2)]  # 转换为OpenCV多边形格式

    point_img = image.copy()
    point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2RGB)
    for point in vertices:
        cv2.circle(point_img, (point[0], point[1]), 5, (0, 0, 255), 2)

    return point_img, cv_vertices

def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)  # 绘制停车区域
    return cv2.bitwise_and(image, mask)

def hough_lines(image):
    # 能够检测出图像中的线段，并返回每条线段的起点和终点坐标
    # image 输入图像，必须是单通道的二值图像（通常是经过边缘检测后的图像，如 Canny 边缘检测的结果
    # rho 直线检测的精度（以像素为单位），默认值：0.1。值越小，检测精度越高
    # theta 直线检测的角度精度（以弧度为单位）。默认值：np.pi / 10（即 18 度）。值越小，检测角度越精细
    # threshold 累加器阈值，用于确定检测到的直线。默认值：15。值越小，检测到的直线越多；值越大，检测到的直线越少
    # minLineLength 线段的最小长度。默认值：9 小于此长度的线段会被忽略
    # maxLineGap 线段之间的最大间隙 默认值：4 如果两条线段之间的间隙小于此值，它们会被合并为一条线段
    return cv2.HoughLinesP(image, rho=0.1, theta=np.pi / 10, threshold=15, minLineLength=9, maxLineGap=4)

def draw_lines(image, lines):
    # 过滤霍夫变换检测到直线
    image = np.copy(image)
    cleaned = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2 - y1) <= 1 and 25 <= abs(x2 - x1) <= 55:
                cleaned.append((x1, y1, x2, y2))
                cv2.line(image, (x1, y1), (x2, y2), [255, 0, 0], 1)
    print("No lines detected: ", len(cleaned))
    return image

def identify_blocks(image, lines):
    _new_image = np.copy(image)

    # Step 1: 过滤部分直线
    cleaned = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2 - y1) <= 1 and 25 <= abs(x2 - x1) <= 55:
                cleaned.append((x1, y1, x2, y2))

    # Step 2: 对直线按照x1进行排序
    import operator
    list1 = sorted(cleaned, key=operator.itemgetter(0, 1))

    # Step 3: 找到多个列，相当于每列是一排车
    clusters = {}
    d_index = 0
    clus_dist = 10

    for i in range(len(list1) - 1):
        distance = abs(list1[i + 1][0] - list1[i][0])
        if distance <= clus_dist:
            if not d_index in clusters.keys():
                clusters[d_index] = []
            clusters[d_index].append(list1[i])
            clusters[d_index].append(list1[i + 1])
        else:
            d_index += 1

    # Step 4: 得到坐标
    rects = {}
    i = 0
    for key in clusters:
        all_list = clusters[key]
        cleaned = list(set(all_list))
        if len(cleaned) > 5:
            cleaned = sorted(cleaned, key=lambda tup: tup[1])
            avg_y1 = cleaned[0][1]
            avg_y2 = cleaned[-1][1]
            avg_x1 = 0
            avg_x2 = 0
            for tup in cleaned:
                avg_x1 += tup[0]
                avg_x2 += tup[2]
            avg_x1 = avg_x1 / len(cleaned)
            avg_x2 = avg_x2 / len(cleaned)
            rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
            i += 1

    print("Num Parking Lanes: ", len(rects))
    # Step 5: 把列矩形画出来
    buff = 7
    for key in rects:
        tup_top_left = (int(rects[key][0] - buff), int(rects[key][1]))
        tup_bot_right = (int(rects[key][2] + buff), int(rects[key][3]))
        cv2.rectangle(_new_image, tup_top_left, tup_bot_right, (0, 255, 0), 1)
    return _new_image, rects

def draw_parking(image, rects, thickness=1, save=False):
    color = [255, 0, 0]
    new_image = np.copy(image)
    gap = 15.5
    cur_len = 0
    spot_dict = {}  # 字典：一个车位对应一个位置
    tot_spots = 0
    # 微调
    adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
    adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}

    adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
    adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}

    for key in rects:
        tup = rects[key]
        # 使用 dict.get 方法，为不存在的键提供默认值 0
        x1 = int(tup[0] + adj_x1.get(key, 0))
        x2 = int(tup[2] + adj_x2.get(key, 0))
        y1 = int(tup[1] + adj_y1.get(key, 0))
        y2 = int(tup[3] + adj_y2.get(key, 0))
        cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        num_splits = int(abs(y2 - y1) // gap)
        for i in range(0, num_splits + 1):
            y = int(y1 + i * gap)
            cv2.line(new_image, (x1, y), (x2, y), color, thickness)
        if 0 < key < len(rects) - 1:
            # 竖直线
            x = int((x1 + x2) / 2)
            cv2.line(new_image, (x, y1), (x, y2), color, thickness)
        # 计算数量
        if key == 0 or key == (len(rects) - 1):
            tot_spots += num_splits + 1
        else:
            tot_spots += 2 * (num_splits + 1)

        # 字典对应好
        if key == 0 or key == (len(rects) - 1):
            for i in range(0, num_splits + 1):
                cur_len = len(spot_dict)
                y = int(y1 + i * gap)
                spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
        else:
            for i in range(0, num_splits + 1):
                cur_len = len(spot_dict)
                y = int(y1 + i * gap)
                x = int((x1 + x2) / 2)
                spot_dict[(x1, y, x, y + gap)] = cur_len + 1
                spot_dict[(x, y, x2, y + gap)] = cur_len + 2

    print(f"total parking spaces: {tot_spots}, len: {cur_len} ")
    if save:
        filename = f'with_parking_{str(random.randint(1000, 9999))}.jpg'
        cv2.imwrite(filename, new_image)
    return new_image, spot_dict

def make_prediction(image, model, class_dictionary):
    # 预处理
    img = image / 255.

    # 转换成4D tensor
    image = np.expand_dims(img, axis=0)

    # 用训练好的模型进行训练
    class_predicted = model.predict(image)
    in_id = np.argmax(class_predicted[0])
    label = class_dictionary[in_id]
    return label

def predict_on_image(image, spot_dict, model, class_dictionary, color=None, alpha=0.5):
    if color is None:
        color = [0, 255, 0]
    new_image = np.copy(image)
    overlay = np.copy(image)
    cnt_empty = 0
    all_spots = 0
    for spot in spot_dict.keys():
        all_spots += 1
        (x1, y1, x2, y2) = spot
        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
        spot_img = image[y1:y2, x1:x2]
        spot_img = cv2.resize(spot_img, (32, 32))

        label = make_prediction(spot_img, model, class_dictionary)
        if label == 'empty':
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            cnt_empty += 1

    cv2.addWeighted(overlay, alpha, new_image, 1 - alpha, 0, new_image)

    cv2.putText(new_image, "Available: %d spots" % cnt_empty, (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.putText(new_image, "Total: %d spots" % all_spots, (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    return new_image


def predict_on_video(video_name, spot_dict, model, class_dictionary, output_name="parking_video_output.mp4"):
    cap = cv2.VideoCapture(video_name)

    # 获取视频帧率、分辨率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置视频写入器（MP4 格式）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    out = cv2.VideoWriter(output_name, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    alpha = 0.5  # 透明度
    color = (0, 255, 0)  # 绿色

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 读取失败，退出循环

        frame_count += 1
        if frame_count % 20 != 0:  # 每 20 帧处理一次，提高性能
            out.write(frame)  # 直接写入原始帧
            continue

        overlay = frame.copy()
        empty_spots = 0
        total_spots = len(spot_dict)

        for (x1, y1, x2, y2) in spot_dict.keys():
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 确保坐标为整数
            spot_img = frame[y1:y2, x1:x2]

            if spot_img.size == 0:
                continue  # 避免无效的裁剪区域

            spot_img = cv2.resize(spot_img, (32, 32))
            label = make_prediction(spot_img, model, class_dictionary)

            if label == 'empty':
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                empty_spots += 1

        # 叠加透明遮罩
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 显示车位信息
        cv2.putText(frame, f"Available: {empty_spots} spots", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {total_spots} spots", (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)  # 将帧写入输出视频文件
        # cv_show('Parking Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_name}")
