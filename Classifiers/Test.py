extract_path = 'D:\DeepLearning\jupyter notebook\Radar\TCABG\DataPreprocessing\Path_to_put_extracted_Test_New Peopledata'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
# random seed.
rand_seed = 1
from numpy.random import seed
seed(rand_seed)
import tensorflow
import tensorflow as tf
tensorflow.random.set_seed(rand_seed)
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from keras.layers.convolutional import *
from tensorflow.keras.layers import Conv3D,Conv2D,MaxPooling2D, MaxPooling3D, LSTM, Dense, Dropout, Flatten, Bidirectional,TimeDistributed,BatchNormalization,LeakyReLU,GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Input, RNN,GRU
from tcn import TCN ,tcn_full_summary
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

test_label = one_hot_encoding(train_label, sub_dirs, categories=6)
test_data = train_data.reshape(train_data.shape[0],train_data.shape[1], train_data.shape[2],train_data.shape[3],1)
del train_data,train_label

print('Testing Data Shape is:')
print(test_data.shape,test_label.shape)

#这里是主要使用的模型结构 CNN_Attention_GRU
def attention_3d_block(inputs,TIME_STEPS,SINGLE_ATTENTION_VECTOR):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = tf.keras.layers.Permute((2, 1))(inputs)
    a = tf.keras.layers.Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = tf.keras.layers.Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1), name='dim_reduction')(a)
        a = tf.keras.layers.RepeatVector(input_dim)(a)
    a_probs = tf.keras.layers.Permute((2, 1), name='attention_vec')(a)
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
        out = self.GRU1(out)
        #(None, None, 128)
        out = attention_3d_block(out, 90, 1)
        out = self.GRU2(out)

        out = self.dropout_2(out)
        out = self.BN(out)
        out = self.dense(out)

        return out


model = MyModel(num_classes=6)
print("Model building is completed")

adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None,
                       decay=0.0, amsgrad=False)

model.compile(run_eagerly=True,loss=keras.losses.categorical_crossentropy,
                   optimizer=adam,
                  metrics=['accuracy'])
tcn_full_summary(model,expand_residual_blocks=True)
model.build(input_shape=(None,90,32,32,1))

model.load_weights("D:\DeepLearning\jupyter notebook\Radar\TCABG\ckpt\model_27-0.99.h5")  # 读取权重，model的结构必须与训练的网络结构一致
model.summary()

#Output ACC
results = model.predict(test_data,batch_size=32,verbose=1)
y_pred = tensorflow.argmax(results, 1) # 预测标签
y_test = tensorflow.argmax(test_label, 1)  # 真实标签
accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(y_pred, y_test), tensorflow.float32))
print("ACC:",accuracy)

# Output confusion matrix
con_mat = confusion_matrix(y_test, y_pred)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # Normalize
con_mat_norm = np.around(con_mat_norm, decimals=2)

label_name = [r'正压腿', r'侧压腿', r'侧边伸展加下拉', r'开合跳', r'扩胸运动', r'关节运动']
from matplotlib import rcParams
def plot_confusion_matrix(con_mat_norm, labels_name):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.heatmap(con_mat_norm, annot=True, cmap='Blues')
    # plt.title(title)    # Plot title
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    num_local = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    plt.xticks(num_local, labels_name, rotation=15)    # Label the x-axis ticks
    plt.yticks(num_local, labels_name, rotation=0)    # Label the y-axis ticks


    plt.ylabel('真实值')
    plt.xlabel('预测值')
    plt.tight_layout()
plot_confusion_matrix(con_mat_norm, label_name)
plt.savefig('D:\\DeepLearning\\jupyter notebook\\Radar\\TCABG\\result\\New People confusion_matrix.png', format='png',dpi=1200)
plt.show()
