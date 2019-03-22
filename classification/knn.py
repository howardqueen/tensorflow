from lib import strMatrix
import numpy as np
import tensorflow as tf

def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("E:/AIML/data/", one_hot=True)
    #mnist = input_data.read_data_sets("/mnt/win10/data/AIML/data", one_hot=True)

    #Xtrain, Ytrain = mnist.train.next_batch(500)  #从数据集中选取5000个样本作为训练集
    Xtest, Ytest = mnist.test.next_batch(20)    #从数据集中选取200个样本作为测试集
    print(strMatrix(Xtest, '', 28, lambda x: 1 if x > 0 else 0));
    return;
    knn_train();

def knn_train():
    print('========================')
    print("= K近邻 - 图片数字识别 =")
    print('========================')    
    Xtrain, Ytrain, Xtest, Ytest = loadData();
    #knn_test(Xtrain, Ytrain, Xtest, Ytest, 2)
    #return
    
    knn = 0.
    max_accuracy = -1
    for k in range(3):
        print('---------------')
        print("训练 K = ", k + 1, "...");
        print()
        accuracy = knn_test(Xtrain, Ytrain, Xtest, Ytest, k + 1)
        print()
        print("完成，准确率", round(accuracy * 100,2), "%")
        if(accuracy > max_accuracy):
            knn = k + 1;
            max_accuracy = accuracy;
    print();
    print('================')
    print('K =', knn, '时准确率(', round(max_accuracy * 100, 2), '%)最高!');
    print('================')
    
    return
    
def loadData():
    print("数据准备...")
    #print('---------------')
    #这里使用TensorFlow自带的数据集作为测试，以下是导入数据集代码
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("E:/AIML/data/", one_hot=True)
    #mnist = input_data.read_data_sets("/mnt/win10/data/AIML/data", one_hot=True)

    Xtrain, Ytrain = mnist.train.next_batch(500)  #从数据集中选取5000个样本作为训练集
    Xtest, Ytest = mnist.test.next_batch(20)    #从数据集中选取200个样本作为测试集
    
    print('---------------')
    print('训练数据, Xtrain:', len(Xtrain), "x", len(Xtrain[0]))
    print('每张图片 784（28x28）个像素点, 比如: ')
    #print(Xtrain[0],end='');
    for i in range(2):
        print('\t[ 图片', i, ']')
        for x in range(28):
            for y in range(28):
                print(1 if Xtrain[i][x*28+y] > 0 else 0, end='');
            print();
    #print(Xtrain)
    print('---------------')
    print('训练标签, Ytrain:', len(Ytrain), "x", len(Ytrain[0]))
    print('共计 10 个分类, 对应 0-9，比如：');
    print('图片0 的标签是', np.argmax(Ytrain[0]), ', 标量是', Ytrain[0]);
    print('图片1 的标签是', np.argmax(Ytrain[1]), ', 标量是', Ytrain[1]);
    #print(Ytrain[1])
    #print("...")
    #print("]")
    print('---------------')
    print('测试数据 Xtest:', len(Xtest), "x", len(Xtest[0]))
    #print('---------------')
    print('测试标签 Ytest:', len(Ytest), "x", len(Ytest[0]))
    return Xtrain, Ytrain, Xtest, Ytest;
    
def knn_test(Xtrain, Ytrain, Xtest, Ytest, k):
    
    # 占位符变量
    xtr = tf.placeholder(tf.float32, [None, 28*28]) # 784=28x28，多张训练集图片，每张图片即一行28x28长度的像素矩阵
    xte = tf.placeholder(tf.float32, [28*28]) # 784=28x28，单张测试图片，该图片即一行28x28长度的像素矩阵
    # 计算L1距离：矩阵横向sum(取每一对abs(xtr-xte))，取图片与图片的像素L1距离
    distance = tf.reduce_sum(
        tf.abs( #取绝对值
            tf.add( #相加
                xtr,
                tf.negative(xte) #取反
            )
        ),
        reduction_indices=1 #矩阵进行横向相加
    )
    
    #分类精确度
    accuracy = 0.
    # 初始化变量
    init = tf.global_variables_initializer()
    # 运行会话，训练模型
    with tf.Session() as sess:

        # 运行初始化
        sess.run(init)

        # 遍历测试数据
        for i in range(len(Xtest)):
            if k <= 1:        
                # 获取当前样本的最近邻索引
                nn_index = sess.run(tf.argmin(distance, 0), feed_dict={xtr: Xtrain, xte: Xtest[i, :]})#向占位符传入训练数据
                ytrmax = np.argmax(Ytrain[nn_index]) # 找出最近邻的标签索引号
                
                ytemax = np.argmax(Ytest[i]) # 找出实际的标签索引号
                # 最近邻标签索引与真实标签索引比较
                if ytrmax == ytemax: # 相等则命中
                    accuracy += 1./len(Xtest)
                    print("TE:np.argmax(", i, Ytest[i], ")=", ytemax,
                          "\t==\tTR:np.argmax(", nn_index, Ytrain[nn_index], ")=", ytrmax)
                else: 
                    print("TE:np.argmax(", i, Ytest[i], ")=", ytemax,
                          "\t<>\tTR:np.argmax(", nn_index, Ytrain[nn_index], ")=", ytrmax, "\t", i, "x")
            else:
                nn_distance = sess.run(distance,  feed_dict={xtr: Xtrain, xte: Xtest[i, :]})# 向占位符传入训练数据
                #nn_distance_transpose = sess.run(tf.transpose(nn_distance))
                nn_items = sess.run(tf.nn.top_k(- nn_distance, k, sorted=True)) # 注意取最短距离，所以距离取反，使得倒序取出
                # print("nn_distance:", nn_distance, "nn_items:", nn_items) #"nn_distance_transpose:", nn_distance_transpose, 
                nn_indexes = nn_items.indices
                #print(nn_items)
                # 找到对应的每个标签
                nn_Ytrain = sess.run(tf.gather(Ytrain, nn_indexes))
                # 找到对应的标签
                nn_Ytrain_vector = []
                for j in range(len(nn_Ytrain)):
                    nn_Ytrain_vector.append(np.argmax(nn_Ytrain[j]))
                # 找到对应每个标签的频次
                uc = sess.run(tf.unique_with_counts(nn_Ytrain_vector))
                uc_index = np.argmax(uc.count) # 最大的频次所在的统计位置
                ytrmax = uc.y[uc_index] # 最大的频次所对应的标签
                nn_index = -1
                for j in range(len(nn_Ytrain)):
                    if np.argmax(nn_Ytrain[j]) == ytrmax:
                        nn_index = nn_indexes[j]
                        break
                if nn_index < 0:
                    print("WTF")
                    return
                    
                ytemax = np.argmax(Ytest[i]) # 找出实际的标签索引号
                # 最近邻标签索引与真实标签索引比较
                if ytrmax == ytemax: # 相等则命中
                    accuracy += 1./len(Xtest)
                    print("TE:np.argmax(", i, Ytest[i], ")=", ytemax, "\t==\tTR:tf.nn.top_k()=",nn_indexes,
                          nn_items.values, nn_Ytrain_vector, "->tf.unique_with_counts()=", uc.y, uc.count,  "->np.argmax()=", ytrmax)
                    #print("TE: np.argmax(", i, ":", Ytest[i], ")=", ytemax, "\t->\tnp.argmax(", nn_index, ":", Ytrain[nn_index], ")=", ytrmax)
                else:
                    print("TE:np.argmax(", i, Ytest[i], ")=", ytemax, "\t<>\tTR:tf.nn.top_k()=",nn_indexes,
                          nn_items.values, nn_Ytrain_vector, "->tf.unique_with_counts()=", uc.y, uc.count,  "->np.argmax()=", ytrmax, "\t", i, "x")
                    #print("<TE: np.argmax(", i, ":", Ytest[i], ")=", ytemax, "\t->\tnp.argmax(", nn_index, ":", Ytrain[nn_index], ")=", ytrmax, "\t->\tx")

    return accuracy

main()
