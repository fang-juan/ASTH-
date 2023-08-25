extract_path = 'D:\DeepLearning\jupyter notebook\Radar\TCABG\DataPreprocessing\Path_to_put_extracted_Test_2m'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
# random seed.
rand_seed = 1
from numpy.random import seed
seed(rand_seed)
import tensorflow
tensorflow.random.set_seed(rand_seed)
tensorflow.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from keras.layers.convolutional import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv3D,Conv2D,MaxPooling3D,MaxPool2D, LSTM, Dense, Dropout, Flatten, Bidirectional,TimeDistributed,GRU,BatchNormalization
from sklearn.model_selection import train_test_split

sub_dirs=['Positive pressure leg','Side pressed leg','Side stretch and pull down','Jumping jacks','Chest enlargement exercise','Joint motion']

def one_hot_encoding(y_data, sub_dirs, categories=6):
    Mapping=dict()

    count=0
    for i in sub_dirs:
        Mapping[i]=count
        count=count+1

    y_features2=[]
    for i in range(len(y_data)):
        Type=y_data[i]
        lab=Mapping[Type]
        y_features2.append(lab)

    y_features=np.array(y_features2)
    y_features=y_features.reshape(y_features.shape[0],1)
    from tensorflow.keras.utils import to_categorical
    y_features = to_categorical(y_features)

    return y_features

frame_tog = [90]

Data_path = extract_path + 'Positive pressure leg'
data = np.load(Data_path + '.npz')
train_data = data['arr_0']
train_data = np.array(train_data, dtype=np.dtype(np.int32))
train_label = data['arr_1']
del data

Data_path = extract_path + 'Side pressed leg'
data = np.load(Data_path + '.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)
del data

Data_path = extract_path + 'Side stretch and pull down'
data = np.load(Data_path + '.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)
del data

Data_path = extract_path + 'Jumping jacks'
data = np.load(Data_path + '.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)
del data

Data_path = extract_path + 'Chest enlargement exercise'
data = np.load(Data_path + '.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)
del data

Data_path = extract_path + 'Joint motion'
data = np.load(Data_path + '.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)
del data

train_label = one_hot_encoding(train_label, sub_dirs, categories=6)
print(train_data.shape)
train_data = train_data.reshape(train_data.shape[0],train_data.shape[1], train_data.shape[2],train_data.shape[3],1)

print('Training Data Shape is:')
print(train_data.shape,train_label.shape)


#划分数据集
X_train, X_val, y_train, y_val  = train_test_split(train_data, train_label, test_size=0.20, random_state=1)
del train_data,train_label
import tensorflow as tf

#这里是主要使用的模型结构 CNN_Attention_GRU
def attention_3d_block(inputs,TIME_STEPS,SINGLE_ATTENTION_VECTOR):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])  #128
    a = tf.keras.layers.Permute((2, 1))(inputs)  # a = (128,90)
    a = tf.keras.layers.Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = tf.keras.layers.Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1), name='dim_reduction')(a)
        a = tf.keras.layers.RepeatVector(input_dim)(a)
    a_probs = tf.keras.layers.Permute((2, 1), name='attention_vec')(a)
    print(a_probs.shape)
    output_attention_mul = tf.keras.layers.Multiply()([inputs, a_probs])
    return output_attention_mul

class MyModel(tensorflow.keras.Model):
    def __init__(self,num_classes = 6,input_shape=(None,32, 32,1)):
        super(MyModel, self).__init__(name = 'my_model')
        self.num_classes = num_classes
        self.input_layer = tensorflow.keras.layers.Input(input_shape)
        #Define my layers here.
        # 1st layer group
        self.Conv3D_1 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv1a", padding="same", activation="relu"))
        self.Conv3D_2 = TimeDistributed(Conv2D(32, (3, 3), strides=(1, 1), name="conv1b", padding="same", activation="relu"))
        Maxp2d1 = tensorflow.keras.layers.MaxPooling2D(name="pool1", strides=(2, 2), pool_size=(2, 2), padding="valid")
        self.MaxPooling3D_1 = TimeDistributed(Maxp2d1)

        self.dropout_1 = Dropout(.5, name='dropout_1')
        self.Flatten =  TimeDistributed(tensorflow.keras.layers.Flatten())

        self.GRU1 = Bidirectional(GRU(64, return_sequences=True, stateful=False))
        self.GRU2 = Bidirectional(GRU(64, return_sequences=False, stateful=False))
        self.dropout_2 = Dropout(.5,name='dropout_2')

        self.BN = BatchNormalization()
        self.dense = tensorflow.keras.layers.Dense(num_classes, activation='softmax', name = 'output')
        self.out = self.call(self.input_layer)

    def call(self,inputs):
        #define my forward pass here,
        #using layers ma previously defined (in "__init__")
        # x = tensorflow.reshape(inputs, [-1, 60, 10240])
        # tf.keras.backend.clear_session()  # 清除之前的模型，省得压满内存
        out = self.Conv3D_1(inputs)
        out = self.Conv3D_2(out)
        out = self.MaxPooling3D_1(out)

        out = self.dropout_1(out)
        out = self.Flatten(out)
        # out = attention_3d_block(out, 90, 1)
        print(out.shape)
        out = self.GRU1(out)
        #(None, None, 128)
        print(out.shape)
        out = attention_3d_block(out, 90, 1)
        out = self.GRU2(out)

        out = self.dropout_2(out)
        out = self.BN(out)
        out = self.dense(out)

        return out

model = MyModel(num_classes=6)

#初始化变量(去掉，否则每次训练的时候都会初始化，导致每个epoch都从头开始，train acc = 20%)
# tensorflow.compat.v1.keras.backend.get_session().run(tensorflow.compat.v1.global_variables_initializer())

print("Model building is completed")
# 显式地构建模型
batch_input_shape = (None, 90,32, 32,1)
model.build(batch_input_shape)
# 计算模型的参数总量
total_params = model.count_params()

print('Total number of parameters: {}'.format(total_params))


# Create a TensorFlow function
def calculate_flops():

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # Run the model and profile the flops
    with tf.compat.v1.Session() as sess:
        flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts).total_float_ops

    return flops


# Calculate the flops
total_flops = calculate_flops()

# Print the flops
print("Total Flops: ", total_flops)
adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None,
                       decay=0.0, amsgrad=False)

model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=adam,
                  metrics=['accuracy'])

filepath="D:\DeepLearning\jupyter notebook\Radar\TCABG\ckpt\model_{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min')

callbacks_list = [checkpoint]

# Training the model
learning_hist = model.fit(X_train, y_train,
                             batch_size=16,
                             epochs=60,
                             verbose=1,
                             shuffle=True,
                           validation_data=(X_val,y_val),
                           callbacks=callbacks_list
                          )
#查看模型
model.summary()

# 绘制图形以确保准确性
# 训练集准确率
plt.plot(learning_hist.history['accuracy'], label='training accuracy')
# 验证集准确率
plt.plot(learning_hist.history['val_accuracy'], label='val accuracy')
plt.title('acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig("D:\\DeepLearning\\jupyter notebook\\Radar\\TCABG\\result\\acc.png",dpi=1200)
plt.legend()
plt.show()

plt.plot(learning_hist.history['loss'], label='training loss')
plt.plot(learning_hist.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig("D:\\DeepLearning\\jupyter notebook\\Radar\\TCABG\\result\\loss.png",dpi=1200)
plt.legend()
plt.show()
