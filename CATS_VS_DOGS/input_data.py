import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_files(file_dir):
    '''
    读取数据和标签
    :param file_dir: 文件路径
    :return: list类型的图片路径和标签
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    file_dir = file_dir + '/'
    for file in os.listdir(file_dir): # 遍历所有文件
        name = file.split(sep='.') # 文件名按.分割
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0) # 如果是猫标签为0
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1) # 如果是狗标签为1

    print('There are %d cats, %d dogs' % (len(label_cats), len(label_dogs)))

    # 将cat和dog的图片和标签整合为列表
    image_list = np.hstack((cats, dogs)) # size: (1, amount)
    label_list = np.hstack((label_cats, label_dogs)) #size: (1, amount)

    temp = np.array([image_list, label_list]) #size: (2, amount)
    temp = temp.transpose() # 转置 (amount, 2)
    np.random.shuffle(temp) # 打乱

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list] # 转换类型

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    分批次获取数据
    :param image: list类型的图片路径
    :param label: list类型的标签
    :param image_W: 图片的宽度
    :param image_H: 图片的高度
    :param batch_size: 每一批的多少张图片
    :param capacity: 队列的最大数量
    :return: 4D的张量 [batch_size, image_W, image_H, 3] tf.float32
             1D的张量 [batch_size] tf.float32
    '''
    #类型转换
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 把image和label合并为一个队列
    input_queue = tf.train.slice_input_producer([image, label])

    # 读取label
    label = input_queue[1]
    image_content = tf.read_file(input_queue[0]) #读取图片
    image = tf.image.decode_jpeg(image_content, channels=3) #解码图片
    #裁剪图片
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # 生成批次
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

image_W = 208
image_H = 208
batch_size = 1
capacity = 256
train_dir = 'dataset_kaggledogvscat/train'

image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, image_W, image_H,
                                     batch_size, capacity)

def get_one_image(train_dir):
    image_list, label_list = get_files(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, image_W, image_H, batch_size, 256)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            if not coord.should_stop():
                img, label = sess.run([image_batch, label_batch])
                if label == 0:
                    print('This is a cat picture!')
                elif label == 1:
                    print('This is a dog picture!')
                #img.astype(np.float32)
                img /= 255
                #plt.imshow(img[0,:,:,:])
                img *= 255
                plt.show()
        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()

        coord.join(threads)
    
    return img
