import numpy as np 
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,\
BatchNormalization,Dropout, Dense, Flatten,AveragePooling2D
from keras.models import Model
from keras.layers import Input, concatenate
from keras.layers import ZeroPadding2D
from keras.layers import add, Flatten

def Conv2d_BN(x, nb_filter, kernel_size, padding ="same", strides=(1,1),name=None):
    if name is not None:
        nb_name = name+"_bn"
        conv_name = name+"_conv"
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter, kernel_size, padding = padding, strides = strides, activation = "relu")(x)
    x = BatchNormalization(axis=3)(x)
    return x

def Conv_Block(inpt, nb_filter, kernel_size, strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1,1), strides=strides, padding ="same")
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3),  padding ="same")
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1),  padding ="same")
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x,shortcut])
    else:
        x= add([x,inpt])
    return x

inpt = Input(shape=(224,224,3))
x = ZeroPadding2D((3,3))(inpt)
x = Conv2d_BN(x, nb_filter=64, kernel_size=(7,7), padding ="valid", strides=(2,2))
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x)

x = Conv_Block(x, nb_filter=[64,64,256],kernel_size=(3,3),strides=(1,1),with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[64,64,256],kernel_size=(3,3))
x = Conv_Block(x, nb_filter=[64,64,256],kernel_size=(3,3))

x = Conv_Block(x, nb_filter=[128,128,512],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[128,128,512],kernel_size=(3,3))
x = Conv_Block(x, nb_filter=[128,128,512],kernel_size=(3,3))
x = Conv_Block(x, nb_filter=[128,128,512],kernel_size=(3,3))

x = Conv_Block(x, nb_filter=[256,256,1024],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[256,256,1024],kernel_size=(3,3))
x = Conv_Block(x, nb_filter=[256,256,1024],kernel_size=(3,3))
x = Conv_Block(x, nb_filter=[256,256,1024],kernel_size=(3,3))
x = Conv_Block(x, nb_filter=[256,256,1024],kernel_size=(3,3))
x = Conv_Block(x, nb_filter=[256,256,1024],kernel_size=(3,3))

x = Conv_Block(x, nb_filter=[512,512,2048],kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
x = Conv_Block(x, nb_filter=[512,512,2048],kernel_size=(3,3))
x = Conv_Block(x, nb_filter=[512,512,2048],kernel_size=(3,3))

x = AveragePooling2D(pool_size=(7,7))(x)
x = Flatten()(x)
x = Dense(1000,activation="softmax")(x)

model = Model(inputs = inpt, outputs=x, name="Resnet-50")

model.summary()