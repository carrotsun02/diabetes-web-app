import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import googleapiclient.discovery
import os
from flask import Flask, render_template
from dotenv import load_dotenv
# 파일 최상단에 이 코드를 추가하세요
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

bootstrap5 = Bootstrap5(app)

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
        # 현재 파일의 위치를 기준으로 절대 경로 생성
        basedir = os.path.abspath(os.path.dirname(__file__))
        csv_path = os.path.join(basedir, 'diabetes.csv')
        model_path = os.path.join(basedir, 'pima_model.h5')

        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])

        # 절대 경로를 사용하여 파일 읽기
        data = pd.read_csv(csv_path, sep=',')

        X = data.values[:, 0:8]
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_test = scaler.transform(X_test)
        
        # 절대 경로를 사용하여 모델 로드
        model = keras.models.load_model(model_path)

        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = (float)(np.round(res * 100, 2)) # 확률을 백분율로 변환

        return render_template('result.html', res=res)
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()