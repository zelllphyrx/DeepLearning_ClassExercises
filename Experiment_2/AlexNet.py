import tensorflow as tf
from d2l import tensorflow as d2l

def net():
    return tf.keras.models.Sequential([
        # 这里使用一个 11*11的更大窗口来捕捉对象
        # 同时，步幅为 4，以减少输出的高度和宽度
        # 另外，输出通道的数目远大于LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 减小卷积窗口，使用填充为 2来让输入与输出的高和宽一致，且增大输出通道数
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 使用3个连续的卷积层和较小的卷积窗口
        # 除了最后的卷积层，输出通道的数量进一步增加
        # 在前两个卷积层之后，池化层不用于减少输入的高度和宽度
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的几倍，使用 dropout层来减轻过度拟合
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # 最后是输出层。由于这里使用 Fashion-MNIST，所以用类别数为 10，而非论文中的 1000
        tf.keras.layers.Dense(10)
    ])

# 测试网络结构
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'Output shape:\t', X.shape)

# 设置批量大小
batch_size = 128

# 加载数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# 定义学习率和迭代次数
lr, num_epochs = 0.01, 10

# 训练模型
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())