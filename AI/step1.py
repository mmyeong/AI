#경사 하강법을 이용한 얕은 신경망 학습
import tensorflow as tf
import numpy as np

## 하이퍼 파라미터 설정
epochs = 1000

## 네트워크 구조 정의
### 얕은 신경망
#### 입력 계층 : 2, 은닉 계층 : 128 (Sigmoid activation), 출력 계층 : 10 (Softmax activation)

# keras의 모듈을 상속해서 Model을 구현
class MyModel(tf.keras.Model):
    def __init__(self):
        # 상속을 한 경우에는 상속을 한 상위 class를 initialize하는 것을 잊어버리지 말자!
        super(MyModel, self).__init__()
        # 아래의 input_dim을 적어줄 필요는 없다. 실제 데이터가 들어올때 정의 되기 떄문이다.
        self.d1 = tf.keras.layers.Dense(128, input_dim=2, activation="sigmoid")
        self.d2 = tf.keras.layers.Dense(10, input_dim=128, activation="softmax")


    # Model이 실제 call이 될때 입력에서 출력으로 어떻게 연결이 될 것인지를 정의
    def call(self, x, training=None, mask=None):
        x = self.d1(x)
        return self.d2(x)

## 학습 루프 정의
@tf.function
# tensorflow의 Auto Graph를 통해 쉽게 구현가능하다.
# function 내의 python 문법으로 입력된 모든 tensor 연산들을 tf.function에 의해서
# 최적화된다.
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):
    # Gradient를 계산하기위한
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(labels, predictions)
    # loss를 model의 trainable_variables(W,b)로 각각 미분해서 gradient를 구한것.
    # loss는 scalar이고, model.trainable_variables는 벡터이므로 결과 또한 벡터가 될 것이다.
    gradients = tape.gradient(loss, model.trainable_variables)

    # 각 gradient와 trainable_variables들이 optimizer로 학습
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # loss를 종합
    train_loss(loss)

    # matric
    train_metric(labels, predictions)

## 데이터셋 생성, 전처리
np.random.seed(0)

pts = []
labels = []

center_pts =  np.random.uniform(-8.0, 8.0, size=(10, 2))
for label, center_pt in enumerate(center_pts):
    for _ in range(100):
        pts.append(center_pt + np.random.randn(*center_pt.shape))
        labels.append(label)

# GPU를 사용하게 된다면 위의 MyModel class에서 initialize 할때
# Layer에 따로 dtype을 지정하지 않으면 float32로 설정되므로 동일하게 해주기 위해 type 재설정
pts =  np.stack(pts, axis=0).astype(np.float32)

# 이미 integer이므로 바꿀 필요가 없음.
labels =  np.stack(labels, axis=0)

# 위에서 만든 데이터를 train data set으로 변형
# train_ds는 iterable한 object가 된다.
# 1000개를 섞어 batch_size를 32개로 해서 구성해준다.
train_ds =  tf.data.Dataset.from_tensor_slices((pts, labels)).shuffle(1000).batch(32)

print(pts.shape)
print(labels.shape)

## 모델 생성
model = MyModel()

## 손실 함수 및 최적화 알고리즘 설정
### CrossEntropy, Adam Optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

## 평가 지표 설정
### Accuracy
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

## 학습 루프
for epoch in range(epochs):

    #위에서 batch_size를 32로 했으므로 한번 실행시 32개씩 나옴.
    for x, label in train_ds:
        train_step(model, x, label, loss_object, optimizer, train_loss, train_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100))

## 데이터셋 및 학습 파라미터 저장
# 압축해서 여러개의 Numpy Object들을 저장할 수 있다.
np.savez_compressed('ch2_dataset.npz', inputs=pts, labels=labels)

W_h, b_h = model.d1.get_weights()
W_o, b_o = model.d2.get_weights()

# weight는 tensorflow에서 사용하고 있는 convention이랑
# shallowNN을 구현할 때 사용했던 convention이 좀 다르다.
W_h = np.transpose(W_h)
W_o = np.transpose(W_o)

np.savez_compressed('ch2_parameters.npz', W_h=W_h, b_h=b_h, W_o=W_o, b_o=b_o)