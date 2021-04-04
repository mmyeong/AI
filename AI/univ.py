import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')
#print(data)

data = data.dropna()

#y데이터 값을 list형태로 가져와줌
yData = data['admit'].values

xData=[]
for i,rows in data.iterrows():
    xData.append([rows['gre'],rows['gpa'],rows['rank']])

#딥러닝 모델 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64,activation='tanh'),
    tf.keras.layers.Dense(128,activation='tanh'),
    tf.keras.layers.Dense(1,activation='sigmoid'),
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(np.array(xData),np.array(yData),epochs=1000)

predict = model.predict([[750,3.70,3],[400,2.2,1]])
print(predict)