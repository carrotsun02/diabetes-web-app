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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

# 경로 설정
basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'pima_model.h5')
csv_path = os.path.join(basedir, 'diabetes.csv')

model = None
scaler = MinMaxScaler()

def initialize_app():
    global model, scaler
    try:
        # 모델 로드
        if os.path.exists(model_path):
            # compile=False로 로드 후 별도 컴파일 (환경 호환성 최적화)
            model = keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("Model loaded successfully from pima_model.h5")
        else:
            print(f"Error: Model file not found at {model_path}")

        # 스케일러 피팅
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            scaler.fit(data.values[:, 0:8])
            print("Scaler initialized successfully")
            
    except Exception as e:
        import traceback
        print(f"Initialization Error: {e}")
        print(traceback.format_exc())

initialize_app()

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

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        if model is None:
            return "Server Error: Model not loaded.", 500
        try:
            input_values = np.array([[
                float(form.preg.data), float(form.glucose.data), float(form.blood.data),
                float(form.skin.data), float(form.insulin.data), float(form.bmi.data),
                float(form.dpf.data), float(form.age.data)
            ]])
            scaled_input = scaler.transform(input_values)
            prediction = model.predict(scaled_input)
            res = float(np.round(prediction[0][0] * 100, 2))
            return render_template('result.html', res=res)
        except Exception as e:
            return f"Prediction Error: {e}", 500
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    # Render 및 클라우드 환경 포트 설정
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)