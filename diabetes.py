import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# 앱 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

# --- [전역 변수 설정] ---
# 현재 파일의 위치를 기준으로 절대 경로 생성
basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'pima_model.keras')
csv_path = os.path.join(basedir, 'diabetes.csv')

model = None
scaler = MinMaxScaler()

# 서버 시작 시 모델과 스케일러를 미리 로드하는 함수
def initialize_app():
    global model, scaler
    try:
        # 1. 모델 로드 (compile=False로 버전 차이로 인한 오류 방지)
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("모델 로드 성공")
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # 2. 스케일러 학습 (데이터 전처리를 위해 CSV 읽기)
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            X = data.values[:, 0:8]
            scaler.fit(X)
            print("스케일러 학습 완료")
        else:
            print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
            
    except Exception as e:
        print(f"초기화 중 에러 발생: {e}")

# 초기화 실행
initialize_app()

# --- [폼 정의] ---
class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF Score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

# --- [라우트 정의] ---
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # 모델 로드 실패 시 예외 처리
        if model is None:
            return "서버 오류: 모델 파일을 로드할 수 없습니다.", 500

        try:
            # 입력 데이터 배열 생성
            input_data = np.array([[
                float(form.preg.data),
                float(form.glucose.data),
                float(form.blood.data),
                float(form.skin.data),
                float(form.insulin.data),
                float(form.bmi.data),
                float(form.dpf.data),
                float(form.age.data)
            ]])

            # 전역 스케일러로 데이터 변환
            X_scaled = scaler.transform(input_data)
            
            # 예측 실행
            prediction = model.predict(X_scaled)
            res = float(np.round(prediction[0][0] * 100, 2))

            return render_template('result.html', res=res)
            
        except ValueError as e:
            return f"입력 값 오류: 숫자를 입력해 주세요. ({e})", 400
        except Exception as e:
            return f"예측 처리 중 오류 발생: {e}", 500
            
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()