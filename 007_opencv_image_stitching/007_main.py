import cv2
import Stitcher
import utils

if __name__ == '__main__':
    image_a = cv2.imread("left_01.png")
    image_b = cv2.imread("right_01.png")

    # 把图片拼接成全景图
    result = Stitcher.stitch(image_a, image_b)
    utils.cv_show("Result", result)
