import os
import numpy as np
import pandas as pd

# [핵심] TensorFlow 메모리 최적화 설정 (가장 먼저 실행되어야 함)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_CPU_FOR_GPUS'] = 'true'

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# CPU 메모리 할당 제한 설정
cpus = tf.config.list_physical_devices('CPU')
if cpus:
    tf.config.set_logical_device_configuration(
        cpus[0], [tf.config.LogicalDeviceConfiguration()]
    )

app = Flask(__name__)
app.config['SECRET_KEY'] = 'assignment-safe-key'
bootstrap5 = Bootstrap5(app)

basedir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(basedir, 'pima_model.h5')
csv_path = os.path.join(basedir, 'diabetes.csv')

# 전역 변수
model = None
scaler = None

def get_ai_resources():
    global model, scaler
    if model is None:
        # 가중치만 가볍게 로드
        model = load_model(model_path, compile=False)
        # 스케일러 초기화
        data = pd.read_csv(csv_path)
        scaler = MinMaxScaler()
        scaler.fit(data.values[:, 0:8])
    return model, scaler

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
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        curr_model, curr_scaler = get_ai_resources()
        try:
            input_data = np.array([[
                float(form.preg.data), float(form.glucose.data), float(form.blood.data),
                float(form.skin.data), float(form.insulin.data), float(form.bmi.data),
                float(form.dpf.data), float(form.age.data)
            ]])
            X_scaled = curr_scaler.transform(input_data)
            prediction = curr_model.predict(X_scaled)
            res = float(np.round(prediction[0][0] * 100, 2))
            return render_template('result.html', res=res)
        except Exception as e:
            return f"Error during prediction: {e}", 500
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)