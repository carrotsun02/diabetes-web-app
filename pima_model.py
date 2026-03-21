import sys
import sklearn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib as mpl

# 버전 확인
assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"
assert tf.__version__ >= "2.0"

# 재현성을 위한 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

# 플롯 설정
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 데이터 로드
data = pd.read_csv('./diabetes.csv', sep=',')
print("\ndata.head(): \n", data.head())

# 데이터 전처리
X = data.values[:, 0:8]
y = data.values[:, 8]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 모델 생성 (Sequential 방식 권장)
model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습
history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

# 학습 결과 시각화
fig, ax1 = plt.subplots()
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color='tab:red')
ax1.plot(history.history['loss'], color='tab:red')
ax2 = ax1.twinx()
ax2.set_ylabel('accuracy', color='tab:blue')
ax2.plot(history.history['accuracy'], color='tab:blue')
plt.show()

# 예측 테스트
X_new = X_test[:3]
print("\nPredict (Before Save): \n", np.round(model.predict(X_new), 2))

# 모델 저장 (H5 포맷 명시)
# 이 파일의 용량이 약 100KB 내외인지 확인하십시오.
model.save('pima_model.h5', save_format='h5')
print("\nModel saved as pima_model.h5")

# 모델 로드 테스트 (메모리상의 모델과 별개로 파일 로드 확인)
reloaded_model = keras.models.load_model('pima_model.h5')
print("\nPredict (After Reload): \n", np.round(reloaded_model.predict(X_new), 2))