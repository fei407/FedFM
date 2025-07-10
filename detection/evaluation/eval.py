import cv2
import torch
from transformers import AutoModelForObjectDetection, AutoProcessor

# 模型路径
model_path = r"C:\Users\fw407\fedfm\object_detection\ffa\peft_100"

# 加载模型和处理器
# model = AutoModelForObjectDetection.from_pretrained("SenseTime/deformable-detr")
model = AutoModelForObjectDetection.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained("SenseTime/deformable-detr")  # 根据需要改为你自己的processor路径

# 设置模型为评估模式
model.eval()

# 输入输出路径
input_image_path = r"C:\Users\fw407\Desktop\detection\image\demo-1.jpg"
output_image_path = r"C:\Users\fw407\Desktop\detection\output\demo-1.jpg"

# 读取图像
image_bgr = cv2.imread(input_image_path)
height, width = image_bgr.shape[:2]
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 图像预处理
inputs = processor(images=image_rgb, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)

# 后处理
results = processor.post_process_object_detection(outputs, threshold=0.2, target_sizes=[(height, width)])[0]

boxes = results["boxes"]
scores = results["scores"]
labels = results["labels"]

def draw_label_with_background(img, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale=0.5, text_color=(255, 255, 255), bg_color=(0, 128, 0)):
    """在图像上绘制带背景的文字"""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
    top_left = (x, y - text_height - 4)
    bottom_right = (x + text_width + 2, y)

    # 背景框
    cv2.rectangle(img, top_left, bottom_right, bg_color, thickness=-1)
    # 文字
    cv2.putText(img, text, (x + 1, y - 2), font, font_scale, text_color, thickness=1, lineType=cv2.LINE_AA)

# 替换原始绘制文字部分
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = map(int, box.tolist())
    class_name = model.config.id2label[label.item()]
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label_text = f"{class_name} {score:.2f}"
    draw_label_with_background(image_bgr, label_text, x1, y1)

# 保存结果
cv2.imwrite(output_image_path, image_bgr)
print(f"检测完成，结果已保存至：{output_image_path}")
