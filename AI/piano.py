import numpy as np
import tensorflow as tf
text= open('pianoabc.txt').read()

#숫자로 바구기
unique_text = list(set(text))
unique_text.sort()

text_to_num={}
num_to_text ={}

for i,data in enumerate(unique_text):
    num_to_text[i]=data
for i,data in enumerate(unique_text):
    text_to_num[data]=i

#원본 텍스트를 모두 숫자로 변환
NumText=[]

for i in text:
    NumText.append(text_to_num[i])

X=[]
Y=[]
for i in range(0,len(NumText)-25):
    X.append(NumText[i:i+25])
    Y.append(NumText[i+25])

X=tf.one_hot(X,31)
Y=tf.one_hot(Y,31)

#모델에 넣기
#LSTM은 epochs를 많이 해줘야함

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100,input_shape=(25,31)),
    tf.keras.layers.Dense(31, activation='softmax')
])#loss를 categorical_crossentropy를 쓰려면 꼭 softmax사용
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,Y,batch_size=64,epochs=10)
model.save('model')