import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt
import numpy as np

# 定义LeNet-5网络结构
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)
    ])

# 加载MNIST数据集
def load_data_mnist(batch_size):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    # 数据预处理
    train_images = tf.expand_dims(train_images, axis=-1)
    test_images = tf.expand_dims(test_images, axis=-1)

    # 确保数据类型一致并归一化
    train_images = tf.cast(train_images, tf.float32) / 255.0
    test_images = tf.cast(test_images, tf.float32) / 255.0

    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int64)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int64)

    # 构建数据迭代器
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
    return train_ds, test_ds

# 加载数据
batch_size = 256
train_iter, test_iter = load_data_mnist(batch_size)

# 定义训练函数，返回训练历史
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr):
    device_name = d2l.try_gpu()._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        net = net_fn()
        net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        history = net.fit(train_iter, epochs=num_epochs, validation_data=test_iter, verbose=1)
    return net, history

# 可视化训练历史
def show_train_history(history, train_metric, val_metric):
    plt.plot(history.history[train_metric], label=f'Train {train_metric}')
    plt.plot(history.history[val_metric], label=f'Validation {val_metric}')
    plt.title(f'Training and Validation {train_metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(train_metric.capitalize())
    plt.legend()
    plt.grid()
    plt.show()

# 可视化预测结果
def plot_images_labels_prediction(images, labels, preds, index=0, num=10):
    # 转换为 NumPy 数组，确保正确处理
    images = images.numpy()
    labels = labels.numpy()
    if preds is not None:
        preds = np.argmax(preds, axis=1)

    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    for i in range(num):
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(images[index].squeeze(), cmap='binary')
        title = f"Label: {labels[index]}"
        if preds is not None:
            title += f"\nPredict: {preds[index]}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1
    plt.show()

# 设置训练参数
lr, num_epochs = 0.9, 10

# 训练模型并返回训练历史
model, train_history = train_ch6(net, train_iter, test_iter, num_epochs, lr)

# 可视化训练损失和准确率
show_train_history(train_history, 'loss', 'val_loss')
show_train_history(train_history, 'accuracy', 'val_accuracy')

# 使用模型对测试集进行预测
test_images, test_labels = next(iter(test_iter))
test_preds = model.predict(test_images)

# 可视化预测结果
plot_images_labels_prediction(test_images, test_labels, test_preds, 0, 10)
