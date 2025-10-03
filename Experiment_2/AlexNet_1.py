import tensorflow as tf
import matplotlib.pyplot as plt
from d2l import tensorflow as d2l

# 定义神经网络
def net():
    return tf.keras.models.Sequential([
        # 第一层卷积：16个3x3卷积核，步幅为1，填充方式为'same'，激活函数为ReLU
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 最大池化：池化窗口大小为2x2，步幅为2
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # 第二层卷积：32个3x3卷积核
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第三层卷积：64个3x3卷积核
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 最大池化
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # 全连接层，展平为一维
        tf.keras.layers.Flatten(),
        # 第层一全连接层：120个神经元
        tf.keras.layers.Dense(120, activation='relu'),
        # Dropout层：丢弃比例为50%
        tf.keras.layers.Dropout(0.5),
        # 第二层全连接层：120个神经元
        tf.keras.layers.Dense(120, activation='relu'),
        # 再次应用Dropout
        tf.keras.layers.Dropout(0.5),
        # 分类类别数为10
        tf.keras.layers.Dense(10)
    ])

batch_size = 128
# 加载Fashion-MNIST数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义学习率和训练的轮数
lr, num_epochs = 1e-3, 10

# 初始化网络、损失函数和优化器
model = net()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 使用交叉熵损失
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # Adam优化器

# 存储训练过程中的损失和准确率
train_loss, train_acc, test_acc = [], [], []

# 定义模型评估函数
def evaluate_accuracy(data_iter, model):
    metric = d2l.Accumulator(2)  # 累计器：正确预测数，总预测数
    for X, y in data_iter:
        y_hat = model(X, training=False)  # 推理模式
        metric.add(d2l.accuracy(y_hat, y), y.shape[0])
    return metric[0] / metric[1]

# 定义训练过程
def train():
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 累计器：损失之和，样本总数，正确预测数
        for X, y in train_iter:
            with tf.GradientTape() as tape:  # 记录梯度
                y_hat = model(X, training=True)  # 前向传播
                l = loss(y, y_hat)  # 计算损失
            grads = tape.gradient(l, model.trainable_variables)  # 计算梯度
            optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 更新参数
            # 更新累计值
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        # 记录当前epoch的损失和准确率
        train_loss.append(metric[0] / metric[2])
        train_acc.append(metric[1] / metric[2])

        # 在测试集上评估准确率
        test_acc_epoch = evaluate_accuracy(test_iter, model)
        test_acc.append(test_acc_epoch)

        # 输出训练进度
        print(f'epoch {epoch + 1}, train loss {train_loss[-1]:.3f}, '
              f'train acc {train_acc[-1]:.3f}, test acc {test_acc_epoch:.3f}')

# 显示网络结构
def print_net_shape():
    X = tf.random.uniform((1, 28, 28, 1))  # 模拟28x28灰度输入
    for layer in model.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'Output shape:', X.shape)

# 训练模型
train()

# 打印网络的输出形状
print_net_shape()

# 绘制训练和测试的结果
def plot_results():
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 绘制训练和测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='train acc')
    plt.plot(epochs, test_acc, label='test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()

# 绘制结果曲线
plot_results()