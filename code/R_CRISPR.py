import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Flatten, LSTM,BatchNormalization, Bidirectional


def ConvBn(inputs,filters, kernel_size, strides=1, padding='same', groups=1):
    x = inputs
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding,groups=groups, use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def RepVGGBlock(inputs, filters, kernel_size, strides=1, padding='same', groups=1, deploy=False):
    x = inputs
    in_channels = inputs.shape[-1]
    rbr_dense = ConvBn(inputs,filters, kernel_size, strides=1, padding='same', groups=1)
    rbr_1x1 = ConvBn(inputs,filters, kernel_size=(1,1), strides=1, padding='same', groups=1)

    if in_channels == filters and strides == 1 :
        rbr_identity = BatchNormalization()(x)
        id_out = rbr_identity
    else:
        id_out = 0

    if deploy:
        rbr_reparam = Conv2D(filters, kernel_size, strides,padding,groups, use_bias=True)(x)
        return tf.nn.relu(rbr_reparam)

    x = tf.nn.relu(rbr_dense + rbr_1x1 + id_out)
    return x


def R_CRISPR_model():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    inputs1=Conv2D(15, (1,1), strides=1, padding='same')(inputs)
    repvgg1 = RepVGGBlock(inputs1, filters=15, kernel_size=(1, 3))

    mixed = Reshape((24, 15))(repvgg1)

    blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(24, 15), name="LSTM_out"))(mixed)
    blstm_out = Flatten()(blstm_out)
    x = Dense(80, activation='relu')(blstm_out)
    x = Dense(20, activation='relu')(x)
    x = tensorflow.keras.layers.Dropout(rate=0.35)(x)

    prediction = Dense(1, activation='sigmoid', name='main_output')(x)
    model = Model(inputs, prediction)
    print(model.summary())
    # 输出各层模型参数情况
    return model


