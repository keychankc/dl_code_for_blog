from pathlib import Path
import pandas as pd

import datetime

import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing

import torch

from nn_weather_forecast.Weather_forecast_NN import Weather_forecast_NN


# 加载数据
def load_csv():
    path = Path("../data/") / "02_temps"
    filename = "temps.csv"
    return pd.read_csv(path / filename)

# 转换日期
def handle_datetime(features):
    years = features['year']
    months = features['month']
    days = features['day']

    return [datetime.datetime(year, month, day) for year, month, day in zip(years, months, days)]

# 绘图
def draw_csv(dates, features):
    # 指定默认风格
    plt.style.use('fivethirtyeight')

    # 设置布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.autofmt_xdate(rotation=45)

    # 当天最高温度（标签值）
    ax1.plot(dates, features['actual'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Max Temp')

    # 昨天最高温度
    ax2.plot(dates, features['temp_1'])
    ax2.set_xlabel('')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Previous Max Temp')

    # 前天最高温度
    ax3.plot(dates, features['temp_2'])
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Two Days Prior Max Temp')

    # 朋友预测最高温度
    ax4.plot(dates, features['friend'])
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Friend Estimate')

    plt.tight_layout(pad=2)
    plt.show()

def get_train_data(features):
    data = features.drop('actual', axis=1)  # 移除真实标签值
    data = np.array(data)  # 转换成合适的格式
    return preprocessing.StandardScaler().fit_transform(data)  # 数据变化，适应训练

def train(train_data, valid_data):
    input_size = train_data.shape[1]
    hidden_size = 128
    output_size = 1
    batch_size = 16
    model = Weather_forecast_NN(input_size, hidden_size, output_size)
    mse_loss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for i in range(1000):
        batch_loss = []
        # MINI-Batch方法来进行训练
        for start in range(0, len(train_data), batch_size):
            end = start + batch_size if start + batch_size < len(train_data) else len(train_data)
            xx = torch.tensor(train_data[start:end], dtype=torch.float32, requires_grad=True)
            yy = torch.tensor(valid_data[start:end], dtype=torch.float32).view(-1, 1)  # 将 target 调整为 (16, 1)
            prediction = model(xx)
            loss = mse_loss(prediction, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.data.numpy())

        # 打印损失
        if i % 100 == 0:
            losses.append(np.mean(batch_loss))
            print(i, np.mean(batch_loss))

    return model

def draw_valid_and_predictions_data(dates, model, train_data, valid_data):

    x = torch.tensor(train_data, dtype=torch.float)
    predict = model(x).data.numpy()
    # 实际值
    plt.plot(dates, valid_data, 'b-', label='actual')
    # 预测值
    plt.plot(dates, predict, 'ro', label='prediction')
    plt.xticks(rotation=60)
    plt.legend()
    # 绘图配置
    plt.xlabel('Date')
    plt.ylabel('Maximum Temperature (F)')
    plt.title('Actual and Predicted Values')
    plt.show()


if __name__ == '__main__':
    # 读取数据
    features = load_csv()
    # 时间处理
    dates = handle_datetime(features)
    # 图像绘制
    # draw_csv(dates, features)
    # 独热编码 不能直接处理类别数据，它们需要数值数据。因此，我们需要将类别特征转换为数值形式
    features = pd.get_dummies(features)
    # print(features.head(5))

    # 训练数据
    train_data = get_train_data(features)
    # 验证数据
    valid_data = np.array(features['actual'])
    # 训练模型
    model = train(train_data, valid_data)
    # 绘制预测值和真实值对比图
    draw_valid_and_predictions_data(dates, model, train_data, valid_data)
