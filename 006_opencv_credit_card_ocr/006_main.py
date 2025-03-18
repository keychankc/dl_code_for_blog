import argparse
import numpy as np
import cv2
import utils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())


# 绘图展示
def cv_show(name, _img):
    cv2.imshow(name, _img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def template_parsing(_template_img):
    # 1️⃣ 颜色转换：灰度化，将原始图片转换为灰度图像（去掉颜色信息）
    # 作用：减少计算量，提高后续处理速度
    ref = cv2.cvtColor(_template_img, cv2.COLOR_BGR2GRAY)
    cv_show('COLOR_BGR2GRAY', ref)

    # 2️⃣ 二值化处理，像素值≤10变为白色（255），像素值>10变为黑色（0）
    # 作用：突出白色数字/字符，便于后续轮廓检测
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    cv_show('THRESH_BINARY_INV', ref)

    # 3️⃣ 轮廓检测，提取二值图中的轮廓
    # RETR_EXTERNAL 只提取外部轮廓，避免嵌套干扰
    # CHAIN_APPROX_SIMPLE 只存储必要的边界点，提高计算效率
    ref_contours, _ = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图上绘制轮廓红色，线宽1px
    # ref_contours:轮廓列表  -1:绘制所有轮廓(0, 0, 255):红色   1:轮廓的线宽
    cv2.drawContours(_template_img, ref_contours, -1, (0, 0, 255), 1)
    cv_show('drawContours', _template_img)

    # 5️⃣ 轮廓排序
    # 作用：按从左到右的顺序排列轮廓，确保数字顺序正确
    ref_contours = utils.sort_contours(ref_contours, method="left-to-right")[0]

    # 6️⃣ 提取单个数字区域
    _digits_dict = {}
    for (i, c) in enumerate(ref_contours):
        (x, y, w, h) = cv2.boundingRect(c)  # 获取最小外接矩形
        roi = ref[y:y + h, x:x + w]  # 截取对应区域，提取数字
        roi = cv2.resize(roi, (57, 88))  # 归一化尺寸，保证所有数字大小一致
        _digits_dict[i] = roi  # 以索引i作为key，以数字模板roi作为value

    # 7️⃣ 显示提取出的数字
    for key, value in _digits_dict.items():
        cv_show(f"Digit {key}", value)

    return _digits_dict  # 用于后续模板匹配


def credit_card_parsing(_card_img):
    # 1️⃣ 将图像转换为灰度图,减少计算复杂度
    gray = cv2.cvtColor(_card_img, cv2.COLOR_BGR2GRAY)
    cv_show('gray', gray)

    # 创建了一个宽度为9，高度为3的矩形结构元素，适用于细长的信用卡号
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    # 2️⃣ 进行顶帽变换（Top-Hat）（先腐蚀再膨胀），增强比背景亮的细小区域
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
    cv_show('tophat', tophat)

    # 3️⃣ 计算水平梯度（Sobel 算子），检测水平方向的变化，适用于信用卡号的横向排列
    grad_x = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)  # 取绝对值
    (minVal, maxVal) = (np.min(grad_x), np.max(grad_x))  # gradX的最大和最小值
    grad_x = (255 * ((grad_x - minVal) / (maxVal - minVal)))  # 将梯度值转换为图像像素值（0 表示黑色，255 表示白色）
    grad_x = grad_x.astype("uint8")  # 类型转8位无符号整型，[0, 255]
    cv_show('gradX1', grad_x)

    # 4️⃣ 通过闭操作（先膨胀，再腐蚀）将数字连在一起，减少破碎字符
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
    cv_show('gradX2', grad_x)

    # 5️⃣ 将图像转换为二值图像（即只有黑白两种像素值），大于阈值的像素设置为 255（白色），小于或等于阈值的像素设置为 0（黑色）
    thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('thresh1', thresh)

    # 6️⃣ 再次闭操作 使用更大的核（5x5）进行闭操作，填补字符内部的小孔洞，确保字符区域完整
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
    cv_show('thresh2', thresh)

    # 7️⃣ 计算轮廓：找到所有白色区域的轮廓
    thresh_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cur_img = _card_img.copy()
    cv2.drawContours(cur_img, thresh_contours, -1, (0, 0, 255), 1)
    cv_show('img', cur_img)

    # 8️⃣ 过滤&选择可能的卡号区域
    _locations = []
    for (i, c) in enumerate(thresh_contours):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)  # 宽高比
        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        if 2.5 < ar < 4.0:
            if (40 < w < 55) and (10 < h < 20):  # 宽 高
                _locations.append((x, y, w, h))
                # 将符合的轮廓从左到右排序
    return gray, sorted(_locations, key=lambda x1: x1[0])
    # [(34, 111, 47, 14), (95, 111, 48, 14), (157, 111, 47, 14), (219, 111, 48, 14)]


def detect_parsing_digits(_locations, _gray_img, _card_img, _digits_dict):
    _output = []  # 存储完整卡号的识别结果
    # 1️⃣ 遍历每个卡号区域
    for (i, (gX, gY, gW, gH)) in enumerate(_locations):
        group_out_put = []  # 存储当前4位数字的识别结果

        # 2️⃣ 提取卡号区域
        group = _gray_img[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        cv_show('group1', group)

        # 3️⃣ 二值化处理，将卡号部分变白，背景变黑，提高对比度
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv_show('group2', group)

        # 4️⃣ 提取每个数字轮廓，确保从左到右排序
        digit_contours, _ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_contours = utils.sort_contours(digit_contours, method="left-to-right")[0]

        # 5️⃣ 遍历每个数字轮廓
        for c in digit_contours:
            # 裁剪出当前数字roi
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            # 调整大小 (57, 88)，以匹配模板字典中的尺寸
            roi = cv2.resize(roi, (57, 88))
            cv_show('roi', roi)

            # 6️⃣ 计算每个数字的匹配得分
            scores = []
            # 在模板中计算每一个得分
            for (digit, digitROI) in _digits_dict.items():
                # 计算ROI与每个模板的匹配程度
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                # 提取最佳匹配分数
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

                # 7️⃣ 选择最佳匹配的数字
            group_out_put.append(str(np.argmax(scores)))

        # 8️⃣ 在原图上绘制识别结果
        cv2.rectangle(_card_img, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(_card_img, "".join(group_out_put), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # 9️⃣ 组合最终的信用卡号码
        _output.extend(group_out_put)
    return _output


if __name__ == '__main__':
    image_path = args["image"]
    template_path = args["template"]
    print(f"image:{image_path},template:{template_path}")

    # 模板图片
    template_img = cv2.imread(template_path)
    cv_show('template_img', template_img)
    digits_dict = template_parsing(template_img)  # 每一个数字对应每一个模板

    # 信用卡图片
    card_img = cv2.imread(image_path)
    card_img = utils.resize(card_img, width=300)
    cv_show('card_img', card_img)
    gray_img, locations = credit_card_parsing(card_img)  # 灰度图 轮廓 从左到右排序

    # 检测解析 数字
    output = detect_parsing_digits(locations, gray_img, card_img, digits_dict)

    # 打印结果
    print("Credit Card #: {}".format("".join(output)))
    cv2.imshow("Image", card_img)
    cv2.waitKey(0)
