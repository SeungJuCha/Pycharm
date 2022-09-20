from SubClassModel import MyModel
import tensorflow as tf

mymodel = MyModel()

mymodel._name = 'subclass_model'

mymodel(tf.keras.layers.Input(shape = (28,28)))
mymodel.summary()
