import cv2
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split

# --- Veri setini yükleme ---
def load_dataset(root_dir):
    image_label_list = []
    for obj_id in sorted(os.listdir(root_dir)):
        obj_folder = os.path.join(root_dir, obj_id)
        if not os.path.isdir(obj_folder):
            continue
        for img_path in glob(os.path.join(obj_folder, '*.png')):
            filename = os.path.basename(img_path)
            if 'l' in filename and 'c' in filename:
                # l ve c arasındaki sayıyı al, örn: "l3c1" -> 3
                light_idx = int(filename.split('l')[1].split('c')[0])
                image_label_list.append((img_path, light_idx - 1))
    return image_label_list

# --- Işık yönü tahmini - gradient ortalaması yöntemi ---
def ldd_gradient_predict(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı: {image_path}")
        return -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Nesne alanını bulmak için eşikleme (nesne siyah değilse)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Nesne piksel konumları
    ys, xs = np.where(thresh > 0)

    if len(xs) == 0 or len(ys) == 0:
        return -1  # nesne yok veya çok karanlık

    # Nesne alanında gradyan hesapla
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Sadece nesne pikselleri için ortalama gradyan
    mean_grad_x = np.mean(grad_x[ys, xs])
    mean_grad_y = np.mean(grad_y[ys, xs])

    # Açıyı hesapla (radyan -> derece)
    angle = np.arctan2(mean_grad_y, mean_grad_x) * 180 / np.pi
    if angle < 0:
        angle += 360

    # Örnek: ışık yönlerini 8 sınıfa bölelim (0-360/8 = 45 derece aralıklar)
    class_idx = int(angle // 45) % 8
    return class_idx

# --- Işık yönü tahmini - value-threshold yöntemi ---
import cv2
import numpy as np
import math

def ldd_value_predict(image_path, num_classes=8):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı: {image_path}")
        return -1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # Nesne alanını bulmak için eşikleme (nesne siyah değilse)
    _, mask = cv2.threshold(v, 10, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return -1

    # Nesne alanındaki parlaklık değerleri
    v_obj = v[ys, xs]

    # Nesne ortalama parlaklığı threshold olarak alıyoruz
    threshold = np.mean(v_obj)

    # Nesne piksellerini aydınlık ve karanlık olarak ayır
    dark_mask = v_obj < threshold
    bright_mask = v_obj >= threshold

    if np.sum(dark_mask) == 0 or np.sum(bright_mask) == 0:
        # Eğer tamamen aydınlık veya tamamen karanlıksa açı hesaplanamaz
        return -1

    # Karanlık bölgenin ağırlıklı merkezi (parlaklık ağırlıklı değil, sadece konum ortalaması)
    dark_xs = xs[dark_mask]
    dark_ys = ys[dark_mask]
    dark_center_x = np.mean(dark_xs)
    dark_center_y = np.mean(dark_ys)

    # Aydınlık bölgenin ağırlıklı merkezi
    bright_xs = xs[bright_mask]
    bright_ys = ys[bright_mask]
    bright_center_x = np.mean(bright_xs)
    bright_center_y = np.mean(bright_ys)

    # Vektör: karanlık merkezden aydınlık merkeze
    vec_x = bright_center_x - dark_center_x
    vec_y = bright_center_y - dark_center_y

    # Açı hesapla (radyan), 0 derece doğuya bakar, saat yönünün tersine pozitif (matematiksel açı)
    angle_rad = math.atan2(-vec_y, vec_x)  # görüntü y ekseni aşağı olduğu için y negatif alınır
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    # Açıyı 8 eşit sınıfa böl (her sınıf 45 derece)
    class_idx = int(angle_deg // (360 / num_classes))
    if class_idx >= num_classes:
        class_idx = num_classes - 1

    return class_idx


# --- Tahminleri karşılaştırıp hata oranını hesaplayan fonksiyon ---
def evaluate_ldd_method(test_list, predict_fn):
    total = len(test_list)
    wrong = 0

    for img_path, true_label in test_list:
        pred_label = predict_fn(img_path)
        if pred_label == -1:
            print(f"Uyarı: Tahmin yapılamadı: {img_path}")
            wrong += 1
            continue
        if pred_label != true_label:
            wrong += 1

    accuracy = (total - wrong) / total * 100
    error_rate = wrong / total * 100
    print(f"Toplam örnek: {total}")
    print(f"Yanlış tahmin sayısı: {wrong}")
    print(f"Doğruluk: %{accuracy:.2f}")
    print(f"Hata oranı: %{error_rate:.2f}")
    return total, wrong, accuracy, error_rate

# --- Ana fonksiyon ---
def main():
    root_dir = 'png4'  # veri klasörü

    # Veri yükle
    full_data = load_dataset(root_dir)
    full_data.sort()

    # Test olarak tüm veriyi al (istersen train/test ayırabilirsin)
    test_list = full_data

    print("=== Gradient Ortalaması Yöntemi Değerlendiriliyor ===")
    evaluate_ldd_method(test_list, ldd_gradient_predict)

    print("\n=== Value Threshold Yöntemi Değerlendiriliyor ===")
    evaluate_ldd_method(test_list, ldd_value_predict)

if __name__ == '__main__':
    main()

