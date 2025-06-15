import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# Model tanımı yeniden yapılmalı
class LightClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(LightClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Tahmin fonksiyonu
def predict_image(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_label = predicted_class.item() + 1  # 1-indexed

    filename = os.path.basename(image_path)
    try:
        light_id = int(filename.split('l')[1].split('c')[0])
        camera_id = int(filename.split('c')[1].split('.')[0])
    except Exception as e:
        light_id, camera_id = None, None
        print("Etiket bilgisi çözümlenemedi:", e)

    print(f"\n📷 Kamera Yönü (cY): {camera_id}")
    print(f"💡 Gerçek Işık Yönü (lY): {light_id}")
    print(f"🔮 Tahmin Edilen Işık Yönü (lY): {predicted_label}")

    arrows = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
    arrow_idx = predicted_label - 1  # çünkü 1-indexed
    arrow = arrows[arrow_idx]
    print(f"🧭 Yön Gösterimi: {arrow}")

    directions = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
    compass = [
        "       ↑       ",
        "   ↖       ↗   ",
        "←     o     →",
        "   ↙       ↘   ",
        "       ↓       "
    ]

    # Gösterimde merkez (o) olan kısmı ok ile değiştir
    arrow_map = {
        0: (2, 12),  # →
        1: (1, 12),  # ↗
        2: (0, 7),   # ↑
        3: (1, 3),   # ↖
        4: (2, 0),   # ←
        5: (3, 3),   # ↙
        6: (4, 7),   # ↓
        7: (3, 12),  # ↘
    }

    row, col = arrow_map[arrow_idx]
    line = list(compass[row])
    line[col] = '*'
    compass[row] = ''.join(line)

    print("\n🧭 ASCII Pusula:")
    for row in compass:
        print(row)

    return predicted_label

# Kullanım
if __name__ == '__main__':
    image_path = '1_l1c1.png'  # Örnek: 'png4/1/1_l3c2.png'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightClassifier(num_classes=8)
    model.load_state_dict(torch.load("light_classifier.pt", map_location=device))
    model.to(device)

    predict_image(model, image_path, transform, device)

