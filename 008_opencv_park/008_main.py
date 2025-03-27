import tensorflow.keras
import parking
import random
import cv2
import train_model
from keras.src.saving import load_model

def keras_version():
    print(tensorflow.keras.__version__)  # 3.9.0

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_parking_area(_image):
    white_yellow_image = parking.select_rgb_white_yellow(_image)
    cv_show("white_yellow_images", white_yellow_image)
    gray_image = parking.convert_gray(white_yellow_image)
    cv_show("gray_image", gray_image)
    edges_image = parking.detect_edges(gray_image)
    cv_show("edges_image", edges_image)
    (point_img, vertices) = parking.select_region(edges_image)
    cv_show("point_img", point_img)  #
    filter_region_image = parking.filter_region(edges_image, vertices)
    cv_show("filter_region_image", filter_region_image)
    lines = parking.hough_lines(filter_region_image)
    line_image = parking.draw_lines(_image, lines)
    cv_show("line_image", line_image)
    rect_image, rects = parking.identify_blocks(_image, lines)
    cv_show("rect_image", rect_image)
    draw_parking_image, _spot_dict = parking.draw_parking(_image, rects)
    cv_show("draw_parking_image", draw_parking_image)
    return _spot_dict

def image_test(_image, _area_dict, _model, _class_dict):
    cv_show("image", _image)
    predicted_image = parking.predict_on_image(_image, _area_dict, _model, _class_dict)
    cv_show("predicted_image", predicted_image)
    filename = f'with_marking_{str(random.randint(1000, 9999))}.jpg'
    cv2.imwrite(filename, predicted_image)

def video_test(video_name, _area_dict, _model, _class_dict):
    parking.predict_on_video(video_name, _area_dict, model, _class_dict)


if __name__ == '__main__':
    # 1.模型训练，生成car1.keras
    train_model.train()
    model = load_model("car1.keras")

    # 2.获取停车场车位区域
    image = cv2.imread("images/frame_0006.jpg")
    area_dict = get_parking_area(image)

    # 3.图片（标出空车位）
    class_dict = {0: 'empty', 1: 'occupied'}
    # image_test(image, area_dict, model, class_dict)

    # 4.视频（标出空车位）
    # video_test('parking_video.mp4', area_dict, model, class_dict)

