import os
from glob import glob
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import torch

# YOLOv5 modelini yükle (sadece person sınıfı)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # person sınıfı

def get_centroid(mask):
    """Binary maskten ağırlık merkezi hesapla."""
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def detect_light_direction(img_path, output_path='output.png'):
    img = cv2.imread(img_path)
    if img is None:
        print(f"HATA: Görüntü yüklenemedi: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    boxes = results.xyxy[0].cpu().numpy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box.astype(int)
        person_crop = img[y1:y2, x1:x2]

        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        v_blur = cv2.GaussianBlur(v, (7, 7), 0)
        v_thresh = cv2.adaptiveThreshold(v_blur, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

        bright_mask = v_thresh
        dark_mask = cv2.bitwise_not(v_thresh)

        bright_center = get_centroid(bright_mask)
        dark_center = get_centroid(dark_mask)

        if bright_center and dark_center:
            cv2.arrowedLine(person_crop, bright_center, dark_center, (0, 0, 255), 2, tipLength=0.3)

        img[y1:y2, x1:x2] = person_crop

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"✔️ Kaydedildi: {output_path}")

# Dataset oluşturma
def load_dataset(root_dir):
    image_label_list = []
    for obj_id in sorted(os.listdir(root_dir)):
        obj_folder = os.path.join(root_dir, obj_id)
        if not os.path.isdir(obj_folder):
            continue
        for img_path in sorted(glob(os.path.join(obj_folder, '*.png'))):
            filename = os.path.basename(img_path)
            if 'l' in filename and 'c' in filename:
                light_idx = int(filename.split('l')[1].split('c')[0])
                image_label_list.append((img_path, light_idx - 1))
    return image_label_list

# Ana script
if __name__ == '__main__':
    root_dir = 'png4'
    output_dir = 'ldd_value_outputs'

    # Aynı test setini yeniden üret
    full_data = load_dataset(root_dir)
    full_data.sort()
    train_list, test_list = train_test_split(full_data, test_size=0.2, random_state=42)

    # Her test görseli için işlem uygula
    for (img_path, label) in test_list:
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        detect_light_direction(img_path, save_path)

