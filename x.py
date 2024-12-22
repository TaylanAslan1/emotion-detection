import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tkinter as tk
from tkinter import messagebox

# Python dosyasının bulunduğu dizin
base_dir = os.path.dirname(os.path.realpath(__file__))

# Dataset dizinlerini Python dosyasının bulunduğu dizine göre ayarlıyoruz
x_dir = os.path.join(base_dir, 'x')  # x klasörü

# train ve test dizinlerini x klasöründen alıyoruz
train_dir = os.path.join(x_dir, 'train')  # Eğitim verisi
test_dir = os.path.join(x_dir, 'test')    # Test verisi

# Model dosyasının yolu
model_path = os.path.join(base_dir, 'emotion_model.h5')

# Görüntü boyutunu ayarlıyoruz (FER-2013 veri setinde 48x48)
img_size = (48, 48)

# Veri artırma (augmentation) ve resim hazırlığı
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    brightness_range=[0.2, 1.0], horizontal_flip=True, 
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test verilerini yüklüyoruz
train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, 
                                                    batch_size=32, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=img_size, 
                                                  batch_size=32, class_mode='categorical')

# Eğer model kaydedilmemişse eğit
if not os.path.exists(model_path):
    # Modeli oluşturuyoruz
    model = Sequential()

    # 1. Convolutional ve MaxPooling katmanları
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2. Convolutional katmanı
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3. Convolutional katmanı
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 4. Batch Normalization ekleme
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 5. Flatten ve Dense katmanları
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Dropout ile aşırı öğrenmeyi engelliyoruz
    model.add(Dense(7, activation='softmax'))  # 7 duygu sınıfı: suprised, sad, neutral, happy, fearful, disgusted, angry

    # Modeli derliyoruz
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # EarlyStopping ile aşırı öğrenmeyi engelliyoruz
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Modeli eğitiyoruz
    model.fit(train_generator, epochs=50, validation_data=test_generator, callbacks=[early_stopping])

    # Modeli kaydediyoruz
    model.save(model_path)
else:
    # Modeli zaten kaydedilmişse, yükle
    model = tf.keras.models.load_model(model_path)

# Etiketler (7 sınıf: suprised, sad, neutral, happy, fearful, disgusted, angry)
labels = ['suprised', 'sad', 'neutral', 'happy', 'fearful', 'disgusted', 'angry']

# Tkinter GUI oluşturma
def start_emotion_detection():
    # Kamerayı açıyoruz
    cap = cv2.VideoCapture(0)

    # Yüz tespiti için Haar Cascade sınıflandırıcıyı yüklüyoruz
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü gri tonlamaya çeviriyoruz
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Yüzleri tespit ediyoruz
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Yüzü kırpıyoruz
            face = frame[y:y+h, x:x+w]

            # Yüzü 48x48 boyutlarına getiriyoruz
            face_resized = cv2.resize(face, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=0)
            face_resized = face_resized / 255.0  # Normalize et

            # Model ile tahmin yapıyoruz
            prediction = model.predict(face_resized)
            max_index = np.argmax(prediction[0])
            emotion = labels[max_index]

            # Yüzün etrafına dikdörtgen çiziyoruz ve duygu yazıyoruz
            color = (0, 255, 0)  # Yeşil renk
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Görüntüyü ekranda gösteriyoruz
        cv2.imshow('Emotion Detection', frame)

        # 'q' tuşuna basarsanız çıkış yapar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kamerayı kapatıyoruz
    cap.release()
    cv2.destroyAllWindows()

# Tkinter penceresini oluşturuyoruz
root = tk.Tk()
root.title("Emotion Detection")

# Başlat butonunu ekliyoruz
start_button = tk.Button(root, text="Start Emotion Detection", command=start_emotion_detection, font=("Arial", 14))
start_button.pack(pady=20)

# Program çalışırken GUI'yi sürekli olarak gösteriyoruz
root.mainloop()
