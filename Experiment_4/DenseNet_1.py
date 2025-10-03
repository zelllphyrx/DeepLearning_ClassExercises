import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

# 定义单个卷积模块
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        # 批量归一化层
        self.bn = tf.keras.layers.BatchNormalization()
        # 激活函数层
        self.relu = tf.keras.layers.ReLU()
        # 卷积层，卷积核大小为3x3，使用相同填充
        self.conv = tf.keras.layers.Conv2D(filters=num_channels,
                                           kernel_size=(3, 3), padding='same')
        # 按顺序堆叠批量归一化、激活和卷积
        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        # 输入数据通过每一层
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        # 最终将输入与特征输出在最后一个维度上进行拼接
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y

# 定义稠密块，由多个卷积模块堆叠
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        # 添加指定数量的卷积模块
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        # 数据依次通过每个卷积模块
        for layer in self.listLayers.layers:
            x = layer(x)
        return x

# 测试稠密块
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))  # 输入为4个8x8的三通道张量
Y = blk(X)  # 稠密块处理数据
Y.shape  # 输出张量形状

# TensorShape([4, 8, 8, 23])

# 定义转换块，用于降维
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        # 批量归一化层
        self.batch_norm = tf.keras.layers.BatchNormalization()
        # 激活函数层
        self.relu = tf.keras.layers.ReLU()
        # 1x1卷积层，用于减少通道数
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        # 平均池化层，用于降低分辨率
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        # 数据通过批量归一化、激活、卷积和池化层
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)

# 测试转换块
blk = TransitionBlock(10)
blk(Y).shape  # 输出张量形状

# TensorShape([4, 4, 4, 10])

# 定义初始卷积块
def block_1():
    return tf.keras.Sequential([
        # 初始卷积层，步幅为2，通道数为64
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),  # 批量归一化
        tf.keras.layers.ReLU(),  # 激活
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')  # 最大池化
    ])

# 定义稠密网络主结构
def block_2():
    net = block_1()  # 添加初始卷积块
    num_channels, growth_rate = 64, 32  # 初始通道数和增长率
    num_convs_in_dense_blocks = [4, 4, 4, 4]  # 每个稠密块包含的卷积模块数量

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))  # 添加稠密块
        num_channels += num_convs * growth_rate  # 更新通道数
        if i != len(num_convs_in_dense_blocks) - 1:  # 若非最后一个稠密块
            num_channels //= 2  # 通道数减半
            net.add(TransitionBlock(num_channels))  # 添加转换块

    return net

# 定义完整网络
def net():
    net = block_2()  # 添加稠密块
    net.add(tf.keras.layers.BatchNormalization())  # 批量归一化
    net.add(tf.keras.layers.ReLU())  # 激活
    net.add(tf.keras.layers.GlobalAvgPool2D())  # 全局平均池化
    net.add(tf.keras.layers.Flatten())  # 展平
    net.add(tf.keras.layers.Dense(10))  # 全连接层，输出10类
    return net

# 设置超参数
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

# 模型编译
model = net()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),  # 优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 损失函数
              metrics=['accuracy'])  # 评价指标

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
