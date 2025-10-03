import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

class Inception(tf.keras.Model):
    """
    GoogLeNet中的Inception块
    'c1'~'c4'分别表示每条路径的输出通道数
    """
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路 1, 单 1 × 1 卷积层
        self.p1_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(c1, 1, activation=None),  # 1x1卷积
            tf.keras.layers.BatchNormalization(),           # 批量归一化
            tf.keras.layers.ReLU()                          # ReLU激活函数
        ])
        # 线路 2, 1 × 1 卷积层后接 3 × 3 卷积层
        self.p2_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(c2[0], 1, activation=None),  # 1x1卷积
            tf.keras.layers.BatchNormalization(),               # 批量归一化
            tf.keras.layers.ReLU()                              # ReLU激活函数
        ])
        self.p2_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(c2[1], 3, padding='same', activation=None),  # 3x3卷积
            tf.keras.layers.BatchNormalization(),                             # 批量归一化
            tf.keras.layers.ReLU()                                            # ReLU激活函数
        ])
        # 线路 3, 1 × 1 卷积层后接 5 × 5 卷积层
        self.p3_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(c3[0], 1, activation=None),  # 1x1卷积
            tf.keras.layers.BatchNormalization(),               # 批量归一化
            tf.keras.layers.ReLU()                              # ReLU激活函数
        ])
        self.p3_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(c3[1], 5, padding='same', activation=None),  # 5x5卷积
            tf.keras.layers.BatchNormalization(),                             # 批量归一化
            tf.keras.layers.ReLU()                                            # ReLU激活函数
        ])
        # 线路 4, 3 × 3 最大池化层后接 1 × 1 卷积层
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')   # 最大池化
        self.p4_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(c4, 1, activation=None),  # 1x1卷积
            tf.keras.layers.BatchNormalization(),           # 批量归一化
            tf.keras.layers.ReLU()                          # ReLU激活函数
        ])

    def call(self, x):
        # 计算每个路径的输出
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度上拼接输出
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])

# 各个模块（已添加批量归一化层）
def b1():
    """
    第一个模块，包含1个卷积层和池化层
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation=None),  # 卷积层
        tf.keras.layers.BatchNormalization(),                                         # 批量归一化
        tf.keras.layers.ReLU(),                                                       # ReLU激活函数
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')            # 最大池化
    ])

def b2():
    """
    第二个模块，包含2个卷积层和池化层
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation=None),  # 1x1卷积
        tf.keras.layers.BatchNormalization(),           # 批量归一化
        tf.keras.layers.ReLU(),                         # ReLU激活函数
        tf.keras.layers.Conv2D(192, 3, padding='same', activation=None),  # 3x3卷积
        tf.keras.layers.BatchNormalization(),           # 批量归一化
        tf.keras.layers.ReLU(),                         # ReLU激活函数
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')  # 最大池化
    ])

def b3():
    """
    第三个模块，包含2个Inception模块和池化层
    """
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')  # 最大池化
    ])

def b4():
    """
    第四个模块，包含5个Inception模块和池化层
    """
    return tf.keras.models.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')  # 最大池化
    ])

def b5():
    """
    第五个模块，包含2个Inception模块和全局平均池化层
    """
    return tf.keras.models.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),  # 全局平均池化
        tf.keras.layers.Flatten()           # 展平
    ])

# 定义GoogLeNet网络结构
def net():
    return tf.keras.Sequential([
        b1(), b2(), b3(), b4(), b5(),
        tf.keras.layers.Dense(10)  # 输出层，10个类别
    ])

# 输出网络的结构
X = tf.random.uniform(shape=(1, 96, 96, 1))  # 随机生成输入数据
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 训练与绘图
lr, num_epochs, batch_size = 0.1, 10, 128  # 学习率、训练轮数和批次大小。这里相对NiN网络没有出现OOM问题，对batch_size不作调整
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)  # 加载数据集

# 编译和训练模型
model = net()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),  # 使用SGD优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 使用交叉熵损失函数
              metrics=['accuracy'])  # 准确率

# 训练模型并保存历史记录
history = model.fit(train_iter, validation_data=test_iter, epochs=num_epochs)

# 训练曲线绘制函数
def plot_training_curves(history):
    """
    绘制训练损失、训练准确率和测试准确率的曲线图。
    参数: history: 训练过程中保存的历史记录，包括训练损失、训练准确率和验证准确率。
    """
    train_loss = history.history['loss']  # 训练损失
    train_acc = history.history['accuracy']  # 训练准确率
    test_acc = history.history['val_accuracy']  # 测试准确率

    epochs = range(1, len(train_loss) + 1)  # 训练的轮次
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 绘制训练损失曲线
    plt.plot(epochs, train_loss, label='train loss', color='blue', marker='o')
    # 绘制训练准确率曲线
    plt.plot(epochs, train_acc, label='train acc', color='green', marker='s')
    # 绘制测试准确率曲线
    plt.plot(epochs, test_acc, label='test acc', color='red', marker='^')

    plt.xlabel('epochs')
    plt.ylabel('metrics')
    plt.title('Training and Validation Metrics')  # 图表标题
    plt.legend()  # 图例
    plt.grid()  # 网格
    plt.show()  # 显示图表

# 调用绘图函数
plot_training_curves(history)
