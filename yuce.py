from ultralytics import YOLO
# 加载训练好的模型，改为自己的路径
model = YOLO('runs/detect/train/weights/best.pt')
# 修改为自己的图像或者文件夹的路径
source = r'D:\openmv-70-_png.rf.314507c67ba1a8647bccf1ec78b2d4a6.jpg' #修改为自己的图片路径及文件名
# 运行推理，并附加参数
model.predict(source, save=True)