import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

# 定义一个VGG块，包含多个卷积层和一个池化层
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()  # 使用Sequential模型按顺序添加层
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,  # 3x3的卷积核
                                   padding='same', activation='relu'))  # 使用ReLU激活函数
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))  # 添加2x2的最大池化层
    return blk

# 定义网络的结构
conv_arch = [(1, 16), (1, 32), (2, 64), (2, 128)]  # 通道数逐步增大

# 定义VGG网络模型
def vgg(conv_arch, input_shape):
    net = tf.keras.models.Sequential()  # 使用Sequential模型按顺序添加层
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))  # 添加每个VGG块
    # 添加全连接层
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),  # 将多维输入展平为一维
        tf.keras.layers.Dense(512, activation='relu'),  # 添加全连接层，输出维度为512
        tf.keras.layers.Dropout(0.5),  # Dropout层，防止过拟合
        tf.keras.layers.Dense(512, activation='relu'),  # 第二个全连接层维度也为512
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)  # 输出层代表10个类别
    ]))
    return net

# 创建一个28×28图像输入的网络
net = vgg(conv_arch, input_shape=(28, 28, 1))

# 加载Fashion-MNIST数据集，并设置批大小
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=28)  # 使用d2l工具加载数据

# 打印网络每层的输出形状
X = tf.random.uniform((1, 28, 28, 1))  # 创建一个28x28的例子
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

# 定义模型编译与训练函数
def compile_and_train(net, train_iter, test_iter, num_epochs, lr):
    model = net()  # 创建模型实例
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # 使用Adam优化器
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 使用交叉熵损失函数
                  metrics=['accuracy'])  # 使用准确率
    history = model.fit(train_iter, epochs=num_epochs, validation_data=test_iter)  # 训练模型，进行验证

    # 提取训练过程中每个epoch的损失和准确率
    epochs = range(1, num_epochs + 1)
    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    test_acc = history.history['val_accuracy']

    # 绘制训练损失
    plt.figure(figsize=(10, 5))  # 图像大小
    plt.subplot(1, 2, 1)  # 左边的子图绘制损失图
    plt.plot(epochs, train_loss, 'r-', label='train loss')  # 损失曲线
    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('Loss')  # y轴标签
    plt.title('Training Loss')  # 图像标题
    plt.legend()  # 图例

    # 绘制训练准确率和测试准确率
    plt.subplot(1, 2, 2)  # 右边子图绘制准确率图
    plt.plot(epochs, train_acc, 'g-', label='train acc')  # 训练准确率曲线
    plt.plot(epochs, test_acc, 'b-', label='test acc')  # 测试准确率曲线
    plt.xlabel('Epochs')  # x轴标签
    plt.ylabel('Accuracy')  # y轴标签
    plt.title('Accuracy')  # 图像标题
    plt.legend()  # 图例

    plt.tight_layout()  # 调整布局，使图像更紧凑
    plt.show()

# 设置训练轮数和学习率
num_epochs, lr = 10, 1e-3

# 调用训练函数并训练模型
compile_and_train(lambda: vgg(conv_arch, input_shape=(28, 28, 1)),
                  train_iter, test_iter, num_epochs, lr)
