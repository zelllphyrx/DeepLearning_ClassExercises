import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

# 定义一个NiN块，添加批量归一化层
def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation=None),  # 不直接激活
        tf.keras.layers.BatchNormalization(),  # 添加批量归一化层
        tf.keras.layers.Activation('relu'),  # 激活函数
        tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation=None),
        tf.keras.layers.BatchNormalization(),  # 添加批量归一化层
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation=None),
        tf.keras.layers.BatchNormalization(),  # 添加批量归一化层
        tf.keras.layers.Activation('relu')
    ])

# 定义整个NiN网络
def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),  # 第一层NiN块
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),  # 最大池化
        nin_block(256, kernel_size=5, strides=1, padding='same'),  # 第二层NiN块
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),  # 最大池化
        nin_block(384, kernel_size=3, strides=1, padding='same'),  # 第三层NiN块
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),  # 最大池化
        tf.keras.layers.Dropout(0.5),  # 随机丢弃，防止过拟合
        # 输出层
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),  # 全局平均池化
        tf.keras.layers.Flatten()  # 展平，形状为(批量大小, 10)
    ])

# 输出网络每层形状
X = tf.random.uniform((1, 224, 224, 1))  # 输入为224x224的灰度图
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 初始化数据及参数
lr, num_epochs, batch_size = 0.1, 10, 64 # 对于出现的GPU显存耗尽问题(Out Of Memory, OOM)，修改batch_size从128到32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# 模型编译
model = net()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型并记录历史数据
history = model.fit(train_iter, epochs=num_epochs, validation_data=test_iter)

# 绘制训练曲线
def plot_training_curves(history):
    # 提取训练和验证的指标数据
    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']

    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))

    # 绘制训练损失曲线
    plt.plot(epochs, train_loss, label='train loss', color='blue', marker='o')
    # 绘制训练准确率曲线
    plt.plot(epochs, train_acc, label='train acc', color='green', marker='s')
    # 绘制测试准确率曲线
    plt.plot(epochs, test_acc, label='test acc', color='red', marker='^')

    plt.xlabel('epochs')
    plt.ylabel('metrics')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.grid()
    plt.show()

# 调用绘图函数
plot_training_curves(history)
