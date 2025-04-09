from ultralytics import YOLO
# 多线程添加代码
if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO(r'C:\Users\28244\Desktop\yolov8\ultralytics\cfg\models\v8\yolov8.yaml')

    # 训练模型
    model.train(data=r'C:\Users\28244\Desktop\yolov8\1.yaml',
                epochs=100,
                batch=16,
                imgsz=640,
                device=0,
                workers=0)