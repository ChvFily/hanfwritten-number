import tensorflow as tf
from tensorflow.keras import datasets
## [60k,28,28]
(x,y),(x_test,y_test) = datasets.mnist.load_data() ##下载数据集

x = tf.convert_to_tensor(x,dtype=tf.float32) / 255  #归一化

y = tf.convert_to_tensor(y,dtype=tf.int32)

x_test = tf.convert_to_tensor(x_test,dtype=tf.float32) / 255  #归一化

y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)
# print(x.shape,y.shape,x.dtype,y.dtype)
# print(x_test.shape,y_test.shape,x_test.dtype,y_test.dtype)
#取数据集
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)  ## 每次从中取出(128,28,28)个train数据
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)  ## 每次从中取出(128,28,28)个test数据
# train_iter = iter(train_db) # 迭代器
# sample = next(train_iter) 
# print("参数：",sample[0].shape,sample[1].shape)
# print(train_db)
#定义参数

""""
x [b,28,28]
y [b]
[b,784] => [b,256] => [b,128] =>[b,10]
w1 [784,256]  w2 [256,128] w3 [128,10]
b1 [256]       b2 [128]     b3 [10]
"""
w1 = tf.Variable(tf.random.truncated_normal([784,256],mean=0,stddev=0.1))
b1 = tf.Variable(tf.zeros(256))
w2 = tf.Variable(tf.random.truncated_normal([256,128],mean=0,stddev=0.1))
b2 = tf.Variable(tf.zeros(128))
w3 = tf.Variable(tf.random.truncated_normal([128,10],mean=0,stddev=0.1))
b3 = tf.Variable(tf.zeros(10))

# print(w1.shape,b1.shape)
# w1.numpy()
# print(b1.shape)

##设置相关参数
lr = 1e-3  ## 学习率 learning rate  
total_correct , total_num = 0 , 0

for iters in range(100): #迭代多次，使技术更加好用的一个方法是什么？
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape: ## 梯度下降函数
            x = tf.reshape(x,[-1,784]) # 转换格式
            # print(x.shape)
            h1 = tf.nn.relu(x@w1+b1) 
            h2 = tf.nn.relu(h1@w2+b2)
            out = h2@w3+b3
            ## compute loss
            ## 1. one hot for y
            y_onehot = tf.one_hot(y,depth=10)
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss) #计算误差的均值
            #梯度下降 更新权值
            grades = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
            w1.assign_sub(lr*grades[0])
            b1.assign_sub(lr*grades[1])
            w2.assign_sub(lr*grades[2])
            b2.assign_sub(lr*grades[3])
            w3.assign_sub(lr*grades[4])
            b3.assign_sub(lr*grades[5])
        if step % 100 == 0:
            print(iters,step,"loss",float(loss))
        
    ## 添加数据测试
    for step,(x_test,y_test) in enumerate(test_db):
        x_t = tf.reshape(x_test,[-1,784]) # 转换格式
        h1 = tf.nn.relu(x_t@w1+b1) 
        h2 = tf.nn.relu(h1@w2+b2)
        out = h2@w3+b3
        prob = tf.nn.softmax(out,axis = 1) ## 概率归一化
        pred = tf.cast(tf.argmax(prob,axis = 1),tf.int32)
        correct = tf.cast(tf.equal(pred,y_test),dtype = tf.int32)
        total_correct_sum = tf.reduce_sum(correct)
        # print(total_sum.numpy())
        total_correct += total_correct_sum ## 实时预测准确率
        total_num += y_test.shape[0]  ## 预测数据
        # print(y_test.numpy())      
    acc = total_correct/total_num
    print("acc:",acc)
    