import tensorflow as tf
import numpy as np
Pmodel = tf.keras.models.load_model('model')
text = open('pianoabc.txt').read()

unique_text = list(set(text))
unique_text.sort()

text_to_num={}
num_to_text={}

for i,data in enumerate(unique_text):
    num_to_text[i]=data
for i,data in enumerate(unique_text):
    text_to_num[data]=i

NumText=[]

for i in text:
  NumText.append(text_to_num[i])


FirstValue=NumText[117:117+25]
FirstValue=tf.one_hot(FirstValue,31)

FirstValue=tf.expand_dims(FirstValue,axis=0)
PredictValue=Pmodel.predict(FirstValue)
np.argmax(PredictValue[0])

print(num_to_text[np.argmax(PredictValue[0])])
print(num_to_text[NumText[117+25]])

music=[]
for i in range(200):
  PredictValue=Pmodel.predict(FirstValue)
  PredictValue=np.argmax(PredictValue[0])

  #음악의 반복이 될 수 있으므로 랜덤도 넣어줍니다.만약 0.9가 나오면 90%확률로 가장 확률이 큰값을 뽑아줍니다.
  #newValue=np.random.choice(unique_text,1,p=PredictValue[0])

  music.append(PredictValue)

  NextInput=FirstValue.numpy()[0][1:]
 # print(NextInput)

  one_hot_num=tf.one_hot(PredictValue,31)
  #print('원핫한거',one_hot_num)

  FirstValue=np.vstack([NextInput,one_hot_num.numpy()])
  FirstValue=tf.expand_dims(FirstValue,axis=0)
print(music)
music_text=[]

for i in music:
  music_text.append(num_to_text[i])
print(music_text)
print(''.join(music_text))