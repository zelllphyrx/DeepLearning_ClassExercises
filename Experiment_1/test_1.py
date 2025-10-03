import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Tensorflow版本：",tf.__version__) # 显示当前Tensorflow版本

# 读取MNIST数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
# 对图像进行归一化处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 将标签转换为one-hot编码
train_labels_ohe = tf.one_hot(train_labels, depth=10)
test_labels_ohe = tf.one_hot(test_labels, depth=10)

# 构建全连接网络模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 将 28x28 图像转换为 784 维向量
    tf.keras.layers.Dense(64, activation=tf.nn.relu),  # 第一个隐藏层，128 个神经元，ReLU 激活函数
    tf.keras.layers.Dense(32, activation=tf.nn.relu),   # 第二个隐藏层，64 个神经元，ReLU 激活函数
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 输出层，10 个神经元，softmax 激活函数
])
# 输出模型摘要
model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
train_history = model.fit(train_images, train_labels_ohe,
                          validation_split=0.2,  # 使用 20% 的训练数据作为验证集
                          epochs=10,  # 训练 10 个周期
                          batch_size=30,  # 每批 32 个样本
                          verbose=2)  # 显示训练过程的详细信息

# 在测试集上评估模型
test_loss, test_accuracy = model.evaluate(test_images, test_labels_ohe, verbose=2)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
train_history.history

def show_train_history(train_history, train_metric, val_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[val_metric])
    plt.title('Train History')
    plt.ylabel(train_metric)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# 调用函数显示训练和验证的损失以及准确率
show_train_history(train_history, 'loss', 'val_loss')
show_train_history(train_history, 'accuracy', 'val_accuracy')

# 使用模型对测试集进行预测，获取模型输出的概率分布
test_pred = model.predict(test_images)

# 可视化预测结果的函数
def plot_images_labels_prediction(images, labels, preds, index=0, num=10):
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    if num > 10:
        num = 10
    for i in range(0, num):
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(np.reshape(images[index], (28, 28)), cmap='binary')
        title = "label=" + str(labels[index])
        if len(preds) > 0:
            title += ", predict=" + str(preds[index])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index = index + 1
    plt.show()

# 调用函数显示预测结果
plot_images_labels_prediction(test_images, test_labels, test_pred, 2, 10)