"""SubClass 장점 모델 내부 레이어의 하이퍼 파라미터 지정가능
    노드 수, num_classes로 랜덤지정 가능"""


import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self,units,num_classes):
        super(MyModel,self).__init__()
        #초기값 설정
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units,activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(units/4, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

        #method overiding
        #훈련용 함수 정의
        # x는 input
    def call(self,x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x