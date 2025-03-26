import numpy as np
import os
from keras import applications, Model
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.layers import Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import optimizers

def files_count(path):
    count = 0
    for entry in os.listdir(path):
        if entry == ".DS_Store":
            continue
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):  # 确保是子目录
            # 使用子目录路径遍历，避免覆盖原始路径
            root, dirs, files = next(os.walk(entry_path))
            count += len(files)
    return count


def load_model(_img_width, _img_height, _num_classes):
    # 加载 VGG16 预训练模型（在 ImageNet 数据集上训练好的权重）
    # include_top=False：去掉 VGG16 的全连接层，只保留卷积部分（用于特征提取）
    # input_shape 3颜色通道
    model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(_img_width, _img_height, 3))

    # 让前10层的参数不参与训练，只使用它们的预训练权重，避免过拟合，并且加快训练速度
    for layer in model.layers[:10]:
        layer.trainable = False  # 冻结VGG16的前10层

    x = model.output  # 获取 VGG16 的输出
    x = Flatten()(x)  # 展平数据（变成一维向量）
    # 添加自定义全连接层  停车/不停车 softmax激活函数：用于分类任务，输出每个类别的概率
    predictions = Dense(_num_classes, activation="softmax")(x)  # 输出层（2 分类）
    # 输入:VGG16的输入层  输出:自定义的分类层predictions
    model = Model(inputs=model.input, outputs=predictions)

    # learning_rate=0.0001 控制学习步长（较小的学习率有助于稳定训练）。
    # momentum=0.9 让优化器记住之前的梯度，减少震荡，提高收敛速度。
    model.compile(loss="categorical_crossentropy",  # 用于多类别分类任务
                  optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),  # 随机梯度下降（SGD）优化器
                  metrics=["accuracy"])  # 训练时监控准确率
    return model


def load_data_generator(_train_data_dir, _valid_data_dir, _img_width, _img_height, _batch_size):
    for directory in [_train_data_dir, _valid_data_dir]:
        ds_store_path = os.path.join(directory, ".DS_Store")
        if os.path.exists(ds_store_path):
            os.remove(ds_store_path)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 归一化到 [0,1]
        horizontal_flip=True,  # 随机水平翻转，适用于左右对称的物体（如汽车、动物）
        fill_mode="nearest",  # 填充方式
        zoom_range=0.1,  # 10% 随机缩放
        width_shift_range=0.1,  # 10% 随机水平平移
        height_shift_range=0.1,  # 10% 随机垂直平移
        rotation_range=5  # 随机旋转 5 度
    )
    # 只进行归一化，不使用数据增强
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # 生成数据集
    train_generator = train_datagen.flow_from_directory(
        _train_data_dir,
        target_size=(_img_height, _img_width),
        batch_size=_batch_size,
        class_mode="categorical"  # 分类模式，适用于多类别（独热编码）
    )
    validation_generator = valid_datagen.flow_from_directory(
        _valid_data_dir,
        target_size=(_img_height, _img_width),
        batch_size=_batch_size,
        class_mode="categorical"
    )
    return train_generator, validation_generator


def train():
    train_data_dir = "data/train"
    valid_data_dir = "data/valid"
    train_files_count = files_count(train_data_dir)
    valid_files_count = files_count(valid_data_dir)
    batch_size = 32
    epochs = 15
    num_classes = 2
    img_width = 32
    img_height = 32

    model = load_model(img_width, img_height, num_classes)
    train_generator, validation_generator = load_data_generator(train_data_dir, valid_data_dir,
                                                                img_width, img_height, batch_size)
    # 训练回调
    # 保存最佳模型
    checkpoint = ModelCheckpoint("car1.keras",
                                 monitor='val_accuracy',  # 监视val_accuracy，如果提高就保存模型
                                 verbose=1,
                                 save_best_only=True,  # 只保存最好的模型，避免保存质量较差的模型
                                 save_weights_only=False,  # 保存整个模型（结构+权重）
                                 mode='auto')  # 自动选择最大/最小模式（val_accuracy 应该是最大化）
    # 如果val_accuracy10轮不变，停止训练
    early = EarlyStopping(monitor='val_accuracy',  # 监视val_accuracy，如果10轮内没有提高，就提前停止训练
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          mode='max')  # val_accuracy 应该最大化

    steps_per_epoch = np.ceil(train_files_count / batch_size).astype(int)
    validation_steps = np.ceil(valid_files_count / batch_size).astype(int)

    model.fit(
        train_generator,  # 训练数据
        steps_per_epoch=steps_per_epoch,  # 每个epoch运行多少个batch（训练批次）
        epochs=epochs,  # 训练 15 轮
        validation_data=validation_generator,  # 验证数据集
        validation_steps=validation_steps,  # 验证批次数
        callbacks=[checkpoint, early]
    )
