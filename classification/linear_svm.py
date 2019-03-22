# Linear Support Vector Machine: Soft Margin
# ----------------------------------
#
# This function shows how to use TensorFlow to
# create a soft margin SVM
#
# We will use the iris data, specifically:
#  x1 = Sepal Length
#  x2 = Petal Width
# Class 1 : I. setosa
# Class -1: not I. setosa
#
# We know here that x and y are linearly seperable
# for I. setosa classification.

from lib import printMartrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

def main():
    loadData();
    lsvm();
    
def lsvm():
    print('=================================')
    print("= 线性支持向量机 - 山鸢尾花判别 =")
    print('=================================')
    x, y = loadData();
    train(x, y);

def loadData():
    print("数据准备...")
    # 加载需要的数据集
    # 加载iris数据集的第一列和第四列特征变量，其为花萼长度和花萼宽度。
    # 加载目标变量时，山鸢尾花为1，否则为-1
    # iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
    iris = datasets.load_iris()
    x_vals = np.array([[x[0], x[3]] for x in iris.data])
    y_vals = np.array([1 if y == 0 else -1 for y in iris.target])
    
    print('---------------')
    print('山鸢尾花的花萼特征(长/宽):');
    printMartrix(x_vals);
    print('---------------')
    print('山鸢尾花的实际判别(是:1/否:-1):');
    printMartrix(y_vals);
    
    return x_vals, y_vals;

def train(x_vals, y_vals):
    # 分割数据集为训练集和测试集
    train_indices = np.random.choice(len(x_vals),
                                     round(len(x_vals)*0.8),
                                     replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)));    
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]

    # 初始化占位符
    x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # Create variables for linear regression
    # 模型变量
    # 对于这个支持向量机算法，我们希望用非常大的批量大小来帮助其收敛。
    # 可以想象一下，非常小的批量大小会使得最大间隔线缓慢跳动。
    # 在理想情况下，也应该缓慢减小学习率，但是这已经足够了。
    # A变量的形状是2×1，因为有花萼长度和花萼宽度两个变量
    A = tf.Variable(tf.random_normal(shape=[2, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    # Declare model operations
    # 声明模型输出
    # 对于正确分类的数据点，如果数据点是山鸢尾花，则返回的数值大于或者等于1；
    # 否则返回的数值小于或者等于-1
    model_output = tf.subtract( # 相减
        tf.matmul( # 矩阵乘法
            x_data,
            A
        ), 
        b
    );

    # Margin term in loss
    classification_term = tf.reduce_mean( # 矩阵各维度求均值
        tf.maximum(
            0., 
            tf.subtract(
                1., 
                tf.multiply( # 简单乘法，对应位置相乘（区别于矩阵乘法）
                    model_output, y_target
                )
            )
        )
    );
    
    # Declare vector L2 'norm' function squared
    # 声明最大间隔损失函数
    # 我们将声明一个函数来计算向量的L2范数。
    # 接着增加间隔参数α。
    # 声明分类器损失函数，并把前面两项加在一起
    l2_norm = tf.reduce_sum( # 矩阵各维度求和
        tf.square( # 平方
            A
        )
    )
    
    # Declare loss function
    # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
    # L2 regularization parameter, alpha
    alpha = tf.constant([0.01])
    # Put terms together
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm));
    # Declare optimizer
    # 声明优化器函数
    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    # Declare prediction function
    # 声明预测函数和准确度函数
    prediction = tf.sign(model_output)
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(prediction, y_target),
            tf.float32
        )
    );
    
    # 初始化模型变量
    init = tf.global_variables_initializer()
    # 创建一个计算图会话
    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        batch_size = 100
        loss_vec = []
        train_accuracy = []
        test_accuracy = []
        for i in range(500):
            debug = True if (i + 1) % 100 == 0 else False;
            if debug:
                print("STEP ",i + 1);
            rand_index = np.random.choice(len(x_vals_train), size=batch_size)
            rand_x = x_vals_train[rand_index]
            rand_y = np.transpose([y_vals_train[rand_index]])
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            if debug:
                printMartrix(rand_index);
                printMartrix(rand_x);
                printMartrix(rand_y);
                print();
            
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            loss_vec.append(temp_loss)

            train_acc_temp = sess.run(accuracy, feed_dict={
                x_data: x_vals_train,
                y_target: np.transpose([y_vals_train])})
            train_accuracy.append(train_acc_temp)

            test_acc_temp = sess.run(accuracy, feed_dict={
                x_data: x_vals_test,
                y_target: np.transpose([y_vals_test])})
            test_accuracy.append(test_acc_temp)

            if debug:
                a = sess.run(A);
                print('Step #{} A=[{},{}], b = {}, Loss={}'.format(
                    str(i+1),
                    str(a[0]),
                    str(a[1]),
                    str(sess.run(b)),
                    str(temp_loss)
                ))

        # 抽取系数
        # 分割x_vals为山鸢尾花（I.setosa）和非山鸢尾花（non-I.setosa）
        [[a1], [a2]] = sess.run(A)
        [[b]] = sess.run(b)
        slope = -a2/a1
        y_intercept = b/a1

        # Extract x1 and x2 vals
        x1_vals = [d[1] for d in x_vals]

        # Get best fit line
        best_fit = []
        for i in x1_vals:
            best_fit.append(slope*i+y_intercept)

        # Separate I. setosa
        setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
        setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
        not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
        not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

def show():
    ops.reset_default_graph()
    
def show1():
    # Plot data and line
    plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
    plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
    plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
    plt.ylim([0, 10])
    plt.legend(loc='lower right')
    plt.title('Sepal Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()

def show2():
    # Plot train/test accuracies
    plt.plot(train_accuracy, 'k-', label='Training Accuracy')
    plt.plot(test_accuracy, 'r--', label='Test Accuracy')
    plt.title('Train and Test Set Accuracies')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def show3():
    # Plot loss over time
    plt.plot(loss_vec, 'k-')
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()
    
main();