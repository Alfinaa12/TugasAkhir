import cv2
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Fungsi untuk ekstraksi fitur warna dari citra
def extract_color_features(image):
    # Konversi citra ke ruang warna HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Ekstraksi fitur warna dari saluran hue dan saturation
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    
    # Hitung nilai rata-rata dan deviasi standar dari hue dan saturation
    mean_hue = np.mean(hue)
    std_hue = np.std(hue)
    mean_saturation = np.mean(saturation)
    std_saturation = np.std(saturation)
    
    # Gabungkan semua fitur menjadi satu vektor fitur
    color_features = [mean_hue, std_hue, mean_saturation, std_saturation]
    
    return color_features

# Path ke folder gambar fresh apple dan stale apple
fresh_apple_folder = "archive/fresh_apple"
stale_apple_folder = "archive/stale_apple"

# Fungsi untuk membaca semua gambar dari folder dan ekstraksi fitur warna
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            img = cv2.imread(path)
            if img is not None:
                images.append((extract_color_features(img), label))
                labels.append(label)
    return images, labels

# Memuat gambar dari folder fresh apple
fresh_images, fresh_labels = load_images_from_folder(fresh_apple_folder, label=0)  # Label 0 untuk mentah

# Memuat gambar dari folder stale apple
stale_images, stale_labels = load_images_from_folder(stale_apple_folder, label=1)  # Label 1 untuk matang

# Menggabungkan data dari kedua kategori
raw_images = fresh_images + stale_images
labels = fresh_labels + stale_labels

# Memecah dataset menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(raw_images, labels, test_size=0.2, random_state=42)

# Extract features and labels separately
X_train_features, X_train_labels = zip(*X_train)
X_test_features, X_test_labels = zip(*X_test)

# Convert the list of feature vectors to a numpy array
X_train_array = np.array(X_train_features)
X_test_array = np.array(X_test_features)

# Inisialisasi model KNN
knn_model = KNeighborsClassifier(n_neighbors=3)

# Latih model KNN
knn_model.fit(X_train_array, y_train)

# Validasi model dengan data uji
y_pred = knn_model.predict(X_test_array)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}%")

# Simpan hasil prediksi ke dalam file CSV
result_df = pd.DataFrame({'True_Label': y_test, 'Predicted_Label': y_pred})
result_df.to_csv('knn_results.csv', index=False)

# Tampilkan hasil prediksi per proses
for i, (example_feature, true_label) in enumerate(zip(X_test_array, y_test)):
    example_feature = example_feature.reshape(1, -1)
    predicted_label = knn_model.predict(example_feature)[0]

    print(f"Proses ke-{i + 1}:")
    print(f"Fitur Contoh: {example_feature}")
    print(f"Label Sebenarnya: {true_label}")
    print(f"Label Prediksi: {predicted_label}\n")

# Jika akurasi di atas 80%, tampilkan hasil prediksi di satu window
if accuracy > 0.8:
    img_path = "hijau_apel.png"  # Ganti dengan path gambar yang ingin diuji
    img = cv2.imread(img_path)

    # Tampilkan citra asli
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Citra Asli")

    # Tampilkan citra grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(3, 3, 2)
    plt.imshow(gray_img, cmap="gray")
    plt.title("Citra Grayscale")

    # Tampilkan citra HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))
    plt.title("Citra HSV")

    # Tampilkan citra R, G, B
    b, g, r = cv2.split(img)
    plt.subplot(3, 3, 4)
    plt.imshow(r, cmap="gray")
    plt.title("Citra R")
    plt.subplot(3, 3, 5)
    plt.imshow(g, cmap="gray")
    plt.title("Citra G")
    plt.subplot(3, 3, 6)
    plt.imshow(b, cmap="gray")
    plt.title("Citra B")

    # ... tambahkan tahap-tahap ekstraksi citra lainnya sesuai kebutuhan ...

    example_features = extract_color_features(img)

    # Prediksi label untuk citra contoh
    predicted_label = knn_model.predict([example_features])

    print("Prediksi Kematangan Buah/Sayur:")
    if predicted_label == 0:
        conclussion = "Matang"
    elif predicted_label == 1:
        conclussion = "Busuk"
    print(conclussion)
        
    # Tampilkan hasil prediksi
    plt.subplot(3, 3, 7)
    plt.text(0.5, 0.5, f"Prediksi:\n{conclussion}", ha="center", va="center", fontsize=12)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

else:
    print("Akurasi model tidak mencukupi. Latih model dengan dataset yang lebih besar.")
