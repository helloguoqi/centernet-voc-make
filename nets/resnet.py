#   -------------------------------------------------------------#
#   ResNet50的网络部分
#   -------------------------------------------------------------#
from __future__ import print_function

from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dropout, MaxPooling2D,
                          ZeroPadding2D)
from tensorflow.keras.regularizers import l2


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 1*1卷积进行通道数调整 通道数缩减
    x = Conv2D(filters1, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 3*3卷积进行特征提取
    x = Conv2D(filters2, kernel_size,padding='same', kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 通道数扩张
    x = Conv2D(filters3, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 通道数调整 下降
    x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer=RandomNormal(stddev=0.02), 
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer=RandomNormal(stddev=0.02), 
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 通道数调整 扩张
    x = Conv2D(filters3, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 残差边部分
    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer=RandomNormal(stddev=0.02), 
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(inputs):
    # 512x512x3
    x = ZeroPadding2D((3, 3))(inputs)
    # 256,256,64
    x = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.02), name='conv1', use_bias=False)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 256,256,64 -> 128,128,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 128,128,64 -> 128,128,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 128,128,256 -> 64,64,512 默认步长为2
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 64,64,512 -> 32,32,1024 默认步长为2
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 32,32,1024 -> 16,16,2048 默认步长为2
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return x

def centernet_head(x,num_classes):
    x = Dropout(rate=0.5)(x)
    #-------------------------------#
    #   解码器
    #-------------------------------#
    num_filters = 256
    # 16, 16, 2048  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64
    for i in range(3):
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    # 最终获得128,128,64的特征层
    # 将预测结果分为三个部分
    # hm header  热力图预测  判断特征点是否指向一个物体及其类别和置信度
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header  中心点预测 获得物体中心距离热力点偏移情况 最后一维度2中的预测值代表当前特征点向右下角偏移情况
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(2, 1, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(y2)

    # reg header  宽高预测 预测值对应预测框的宽高
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4))(y3)
    return y1, y2, y3
