import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class SplicedHumanDataset:
    def __init__(self, image_dir, mask_dir, threshold=0.5, weights_path='weights/yolov8n-seg.pt', device='cpu'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.threshold = threshold
        self.device = device

        self.model = YOLO(weights_path)
        self.model.to(device)

        self.first_show = True

    def crop_with_mask(self, image_pil, mask_np):
        image_np = np.array(image_pil)
        h, w = mask_np.shape

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = image_np
        rgba[..., 3] = mask_np * 255

        return Image.fromarray(rgba, mode='RGBA')

    def resize_mask_to_image(self, mask_np, target_size):
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_resized = mask_img.resize(target_size, resample=Image.NEAREST)
        mask_resized_np = np.array(mask_resized) // 255
        return mask_resized_np.astype(np.uint8)

    def detect_humans_and_background(self, image_pil, mask_pil):
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil.convert('L'))

        h_img, w_img = image_np.shape[:2]

        results = self.model(image_np)

        persons_crops = []
        labels = []
        masks = []

        for r in results:
            class_ids = r.boxes.cls.cpu().numpy()
            for idx_mask, cls_id in enumerate(class_ids):
                if cls_id == 0:
                    yolo_mask_raw = r.masks.data[idx_mask].cpu().numpy().astype(np.uint8)

                    yolo_mask = self.resize_mask_to_image(yolo_mask_raw, (w_img, h_img))
                    masks.append(yolo_mask)

                    real_mask_area = mask_np[yolo_mask == 1]
                    white_ratio = np.mean(real_mask_area == 255) if len(real_mask_area) > 0 else 0
                    label = 1 if white_ratio > self.threshold else 0
                    labels.append(label)

                    cropped_person = self.crop_with_mask(image_pil, yolo_mask)
                    persons_crops.append(cropped_person)

        if len(masks) == 0:
            background_crop = image_pil.convert('RGBA')
        else:
            combined_mask = np.zeros_like(masks[0])
            for m in masks:
                combined_mask = np.maximum(combined_mask, m)
            background_mask = 1 - combined_mask
            background_crop = self.crop_with_mask(image_pil, background_mask)

        if self.first_show:
            self.show_crops(persons_crops, labels, background_crop)
            self.first_show = False

        return persons_crops, labels, background_crop

    def show_crops(self, persons, labels, background):
        n = len(persons)
        plt.figure(figsize=(4*(n+1), 4))

        for i, (p, l) in enumerate(zip(persons, labels)):
            plt.subplot(1, n+1, i+1)
            plt.imshow(p)
            plt.title(f"Person {i+1} - Label: {l}")
            plt.axis('off')

        plt.subplot(1, n+1, n+1)
        plt.imshow(background)
        plt.title("Background")
        plt.axis('off')

        plt.show()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')

        return self.detect_humans_and_background(image, mask_img)


if __name__ == "__main__":
    dataset = SplicedHumanDataset(
        image_dir='image',
        mask_dir='mask',
        threshold=0.5,
        device='cpu'
    )

    sample = dataset[0]
    print(f"Kişi sayısı: {len(sample[0])}")
    print(f"Etiketler: {sample[1]}")
