import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

class Residual(tf.keras.Model):  # Save
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        """
        残差模块的初始化函数

        参数：
        - num_channels:卷积层输出的通道数
        - use_1x1conv:是否使用1x1卷积调整输入的形状
        - strides:卷积层的步幅
        """
        super().__init__()
        # 第一层卷积：3x3卷积，用于提取特征
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        # 第二层卷积：3x3卷积，用于进一步特征提取
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        # 可以添加1x1卷积层调整输入形状
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        # 批归一化层，用于加速训练
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        """
        前向传播函数

        - 输入X通过第一层卷积和批归一化后进行激活
        - 第二层卷积后与输入X进行残差连接
        - 输出经过激活函数后返回
        """
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))  # 第一卷积 + 批归一化 + 激活
        Y = self.bn2(self.conv2(Y))  # 第二卷积 + 批归一化
        if self.conv3 is not None:  # 如果需要调整输入形状
            X = self.conv3(X)
        Y += X  # 残差连接
        return tf.keras.activations.relu(Y)  # 激活后输出

# 测试Residual模块
blk = Residual(3)  # 创建一个残差模块，不调整输入形状
X = tf.random.uniform((4, 6, 6, 3))  # 随机生成输入数据
Y = blk(X)  # 前向传播
print(Y.shape)  # 输出形状

blk = Residual(6, use_1x1conv=True, strides=2)  # 使用1x1卷积和步幅调整输入形状
print(blk(X).shape)  # 输出形状

# 定义ResNet的初始块，包括卷积、批归一化和池化
blk = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),  # 初始大卷积核
    tf.keras.layers.BatchNormalization(),  # 批归一化
    tf.keras.layers.Activation('relu'),  # 激活函数
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')  # 最大池化，进一步缩小特征图尺寸
])

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        """
        ResNet的残差块初始化

        参数：
        - num_channels:每个残差模块的输出通道数
        - num_residuals:残差模块的数量
        - first_block:是否为网络的第一块残差块
        """
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                # 在非第一块的第一个残差模块中调整输入形状
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                # 其他残差模块不改变输入形状
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        """
        前向传播函数：顺序通过残差模块
        """
        for layer in self.residual_layers:
            X = layer(X)
        return X

# 构建ResNet的不同残差块
b2 = ResnetBlock(64, 2, first_block=True)  # 第一块残差块
b3 = ResnetBlock(128, 2)  # 第二块
b4 = ResnetBlock(256, 2)  # 第三块
b5 = ResnetBlock(512, 2)  # 第四块

# 之前定义一个函数，用它在 'tf.distribute.MirroredStrategy' 的范围
# 来利用各种计算资源，例如 GPU。另外，尽管已经创建了 b1. b2. b3. b4. b5，
# 但是将在这个函数的作用域内重新创建它们

# 定义ResNet网络
def net():
    """
    定义ResNet网络，包括初始卷积块、多个残差块和全连接层
    """
    return tf.keras.Sequential([
        # 初始卷积块
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # 多个残差块
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        # 全局平均池化和全连接层
        tf.keras.layers.GlobalAvgPool2D(),  # 全局平均池化，转换为一维
        tf.keras.layers.Dense(units=10)  # 输出层，分类为10类
    ])

# 打印每一层的输出形状
X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 模型训练配置
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)  # 加载Fashion-MNIST数据集

# 模型编译
model = net()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),  # 优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 损失函数
              metrics=['accuracy'])  # 评价指标

# 模型训练
history = model.fit(train_iter, epochs=num_epochs, validation_data=test_iter)

# 绘制训练曲线
def plot_training_curves(history):
    """
    绘制训练和验证的损失、准确率曲线
    """
    train_loss = history.history['loss']  # 提取训练损失
    train_acc = history.history['accuracy']  # 提取训练准确率
    test_acc = history.history['val_accuracy']  # 提取验证准确率

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
