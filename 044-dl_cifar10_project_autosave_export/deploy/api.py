"""使用 FastAPI 封装一个简单的在线推理服务。

接口说明：
- POST /predict
    - form-data 中上传字段名为 `file` 的图片
    - 返回 JSON: {"prediction": <int>}

启动服务：
    uvicorn deploy.api:app --host 0.0.0.0 --port 8000 --reload

然后可以用 curl / Postman 等工具测试：
    curl -X POST -F "file=@test.jpg" http://localhost:8000/predict
"""

from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from torchvision import transforms


app = FastAPI(title="CIFAR10 SimpleCNN API")

# 全局模型变量，在应用启动时加载一次，避免每次请求重复加载
model = None

# 与训练大致一致的预处理流程
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


@app.on_event("startup")
def load_model():
    """在 FastAPI 服务启动时加载一次 TorchScript 模型。"""

    global model
    model_path = "model.pt"  # 默认模型路径，可以根据需要改成绝对路径
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    print(f"[INFO] 模型已加载: {model_path}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """接收一张图片并返回预测类别。"""

    # 从上传的文件中读取图片，并转成 RGB
    img = Image.open(file.file).convert("RGB")

    # 做与训练时一致的预处理
    x = transform(img).unsqueeze(0)  # [1, 3, 32, 32]

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    return {"prediction": int(pred)}


# 也可以像普通脚本一样直接运行本文件启动服务：
#   python deploy/api.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
