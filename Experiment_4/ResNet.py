import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

class Residual(tf.keras.Model):  # Save
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)

blk = Residual(3)
X = tf.random.uniform((4, 6, 6, 3))
Y = blk(X)
Y.shape

# TensorShape([4, 6, 6, 3])

blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape

# TensorShape([4, 3, 3, 6])

blk = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
])

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X

b2 = ResnetBlock(64, 2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)

# 之前定义一个函数，用它在 'tf.distribute.MirroredStrategy' 的范围
# 来利用各种计算资源，例如 GPU。另外，尽管已经创建了 b1. b2. b3. b4. b5，
# 但是将在这个函数的作用域内重新创建它们

def net():
    return tf.keras.Sequential([
        # 以下各层与之前创建的 b1 层相同
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # 以下各层与之前创建的 b2、b3、b4 和 b5 相同
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)
    ])

X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

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
