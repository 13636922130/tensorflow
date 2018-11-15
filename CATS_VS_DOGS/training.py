import tensorflow as tf
import input_data
import model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMG_W = 208
IMG_H = 208
BATCH_SIZE = 32
CAPACITY = 256
N_CLASSES = 2
LEARNING_RATE = 0.001
MAX_STEP = 10000
DISPLAY_STEP = 50
SAVE_STEP = 200

istrain = 1
ckpt_dir = './ckpt'
train_dir = './dataset_kaggledogvscat/train'
test_dir = './dataset_kaggledogvscat/test'

def run_training():
    
    step_plot = []
    cost_plot = []
    accu_plot = []

    train, train_label = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H,BATCH_SIZE, CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, LEARNING_RATE)
    train_acc = model.evaluation(train_logits, train_label_batch)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        if istrain == 1:
            print('Start training...')
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
    

                if step % DISPLAY_STEP == 0:
                    step_plot.append(step)
                    cost_plot.append(tra_loss)
                    accu_plot.append(tra_acc)
                    print('step %d, train loss is %.2f, train accuracy is %.2f'
                        % (step, tra_loss, tra_acc * 100))

                if step % SAVE_STEP == 0:
                    saver.save(sess, ckpt_dir + '/model.ckpt', global_step=step)
        else:
            flag = 0
            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success,  step is %s' % start_step)
                flag = 1
            else:
                print('No checkpoint file found')
            
            if flag == 1:
                for step in range(int(start_step), MAX_STEP):
                    if coord.should_stop():
                        break
                    _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
            
                    if step % DISPLAY_STEP == 0:
                        print('step %d, train loss is %.2f, train accuracy is %.2f'
                            % (step, tra_loss, tra_acc * 100))

                    if step % SAVE_STEP == 0:
                        saver.save(sess, ckpt_dir + '/model.ckpt', global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training')

    finally:
        coord.request_stop()

    saver.save(sess, ckpt_dir + '/model.ckpt', global_step=step)
    plt.title('Result Analysis')
    plt.plot(step_plot, cost_plot, label='loss')
    plt.plot(step_plot, accu_plot, label='accuracy')
    plt.legend()
    plt.xlabel('Iteration times')
    plt.ylabel('Rate')
    plt.yticks([])
    plt.show()
    plt.savefig('/home/cyq/Desktop/CatsVsDogs.png')

    coord.join(threads)
    sess.close()


def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image():
    image = input_data.get_one_image(train_dir)
    #image = tf.cast(image, tf.float32)
    '''
    image_list, label_list = input_data.get_files(train_dir)
    image_array = get_one_image(image_list)
    print(image_array)
    image = tf.cast(image_array, tf.float32)
    image = tf.reshape(image, [1, 208, 208, 3])
    print(type(image[0][0][0][0]), image.shape)
    '''
    logit = model.inference(image, 1, 2);
    logit = tf.nn.softmax(logit)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Reading checkpoints...')
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Load the weight of the step', step)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success!')
        else:
            print('No checkpoint file found!')

        pre = sess.run(logit)
        max_index = np.argmax(pre)

        if max_index == 0:
            print('\nI think this is a cat with possibility %.6f%%\n' % (pre[:, 0] * 100))
        else:
            print('\nI think this is a dog with possibility %.6f%%\n' % (pre[:, 1] * 100))
        image /= 255
        plt.imshow(image[0,:,:,:])
        plt.show()

#run_training()
evaluate_one_image()
