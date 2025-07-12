import cv2
import torch
import time
from transformers import AutoModelForObjectDetection, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device type: {device}")

# base model & fine-tuned model
# model_path = "SenseTime/deformable-detr"
# model_path = r"C:\Users\fw407\fedfm\object_detection\lr1e-3"
model_path = r"C:\Users\fw407\fedfm\object_detection\lr5e-4"
# model_path = r"C:\Users\fw407\fedfm\object_detection\lr1e-4"

model = AutoModelForObjectDetection.from_pretrained(model_path)

processor = AutoProcessor.from_pretrained("SenseTime/deformable-detr")
model = model.to(device)
model.eval()

if "SenseTime" in model_path:
    threshold = 0.4
    print("Using base model, threshold = 0.4")
else:
    threshold = 0.3
    print("Using finetuned model, threshold = 0.3")

# 输入输出路径
input_video_path = r"C:\Users\fw407\Desktop\detection\video\running_woman.mp4"
output_video_path = r"C:\Users\fw407\Desktop\detection\output\running_woman.mp4"

# 打开视频
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def draw_label_with_background(img, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale=0.5, text_color=(255, 255, 255), bg_color=(0, 128, 0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
    top_left = (x, y - text_height - 4)
    bottom_right = (x + text_width + 2, y)
    cv2.rectangle(img, top_left, bottom_right, bg_color, thickness=-1)
    cv2.putText(img, text, (x + 1, y - 2), font, font_scale, text_color, thickness=1, lineType=cv2.LINE_AA)

frame_count = 0
total_infer_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 图像预处理
    b, g, r = cv2.split(frame)
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    frame_eq = cv2.merge([b_eq, g_eq, r_eq])
    frame_rgb = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)

    # 模型输入
    inputs = processor(images=frame_rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理计时
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    infer_time = (time.time() - start_time) * 1000
    total_infer_time += infer_time

    # 后处理
    results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=[(height, width)])[0]
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = model.config.id2label[label.item()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_label_with_background(frame, f"{class_name} {score:.2f}", x1, y1)

    out.write(frame)
    frame_count += 1
    print(f"Frame {frame_count}: inferecnce time = {infer_time:.2f} ms")

# 平均耗时
avg_time = total_infer_time / frame_count if frame_count else 0
print(f"\nAverage processing time per frame: {avg_time:.2f} ms")

cap.release()
out.release()
print(f"Annotation completed, results are saved at: {output_video_path}")
