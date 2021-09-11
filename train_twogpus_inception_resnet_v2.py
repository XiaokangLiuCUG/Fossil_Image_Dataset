import os
import numpy as np
from time import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_resnet_v2 as inception_resnet_v2
from tensorflow.python import pywrap_tensorflow

import input_data
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

num_gpu = 2
START_LEARNING_RATE = 0.0002

DECAY_RATE = 0.96
BATCH_SIZE = 50 * num_gpu
N_CLASSES = 50
HEIGHT = 299
WIDTH = 299
epochs = 40
each_epoch_step = int(332029/BATCH_SIZE+1)
DECAY_STEPS = int(each_epoch_step/2)
MAX_STEPS = each_epoch_step*epochs
MOVING_AVERAGE_DECAY = 0.99
#test parameters
n_test = 20782
num_batch = int(n_test / BATCH_SIZE+1)
num_sample = num_batch*BATCH_SIZE

train_log_dir = "D:/xiaokang/macrofossil-data/netslogs/inception_resnet_v2_gpu2_67decay0_5crop/save_model/"
test_log_dir = "D:/xiaokang/macrofossil-data/netslogs/inception_resnet_v2_gpu2_67decay0_5crop/validation_log/"
train_data_dir = 'F:/xiaokang/macrofossiltfrecord/train/traindata.tfrecords*'
validation_data_dir = 'F:/xiaokang/macrofossiltfrecord/validation\\validationdata.tfrecords*'
test_data_dir = 'F:/xiaokang/macrofossiltfrecord/test/testdata.tfrecords*'


pre_trained_weights = 'C:/Users/DELL/Jupyter_code/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt'

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionResnetV2/AuxLogits/Logits,InceptionResnetV2/Logits/Logits'
TRAINABLE_SCOPES = 'InceptionResnetV2/Mixed_6a,InceptionResnetV2/Repeat_1,InceptionResnetV2/Mixed_7a,InceptionResnetV2/Repeat_2,InceptionResnetV2/Block8,InceptionResnetV2/Conv2d_7b_1x1,InceptionResnetV2/AuxLogits/Logits,InceptionResnetV2/Logits/Logits'

def average_losses(loss,global_step):
    tf.add_to_collection('losses', loss)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses')

    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step, name='avg')
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        if grads[0] is not None:
            #print("grads",grads)
            grad = tf.stack(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads

def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y):
    for i in range(len(models)):
        x, y, _, _, _ = models[i]
        start_pos = int(i * payload_per_gpu)
        stop_pos = int((i + 1) * payload_per_gpu)
        inp_dict[x] = batch_x[start_pos:stop_pos]
        inp_dict[y] = batch_y[start_pos:stop_pos]
    return inp_dict

def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(",")]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            with tf.variable_scope(exclusion):
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(",")]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def main(argv=None):
    tf.reset_default_graph()
    with tf.Session() as sess:
        startTime =time()
        with tf.device('/cpu:0'):
            #read train data
            training_images, training_labels = input_data.read_TFRecord(data_dir=train_data_dir,
                                                     batch_size= BATCH_SIZE,
                                                     shuffle=True,
                                                     in_classes=N_CLASSES,
                                                     IMG_HEIGHT=HEIGHT,
                                                     IMG_WIDTH=WIDTH,
                                                     is_training=True)
            #read validation data
            validation_images, validation_labels = input_data.read_TFRecord(data_dir=validation_data_dir,
                                                     batch_size= BATCH_SIZE,
                                                     shuffle=True,
                                                     in_classes=N_CLASSES,
                                                     IMG_HEIGHT=HEIGHT,
                                                     IMG_WIDTH=WIDTH,
                                                     is_training=False)

            #read test data
            test_images, test_labels = input_data.read_TFRecord(data_dir=test_data_dir,
                                                     batch_size= BATCH_SIZE,
                                                     shuffle=False,
                                                     in_classes=N_CLASSES,
                                                     IMG_HEIGHT=HEIGHT,
                                                     IMG_WIDTH=WIDTH,
                                                     is_training=False)
            keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            learning_rate = tf.train.exponential_decay(START_LEARNING_RATE,global_step,DECAY_STEPS,DECAY_RATE, staircase=True) 
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            print('build model...')
            print('build model on gpu tower...')
            models = []
            for gpu_id in range(num_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower:%d...'% gpu_id)
                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('', reuse=gpu_id>0):
                            x = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='input_images')
                            y = tf.placeholder(tf.int64, shape=[None, N_CLASSES], name='labels') 
#                             with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
#                                 pred, _ = inception_v4.inception_v4(x, num_classes=N_CLASSES,is_training=True,dropout_keep_prob=keep_prob)
                            with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                                pred, _ = inception_resnet_v2.inception_resnet_v2(x, num_classes=N_CLASSES, is_training=True,dropout_keep_prob=keep_prob)
                            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
                            ave_loss = average_losses(loss,global_step)
                            trainable_variables = get_trainable_variables()
                            grads = opt.compute_gradients(ave_loss,var_list=trainable_variables)#tf.trainable_variables()
                            models.append((x,y,pred,loss,grads))
            print('build model on gpu tower done.')

            print('reduce model on cpu...')
            tower_x, tower_y, tower_preds, tower_losses, tower_grads = zip(*models)
            #print("get_trainable_variables",trainable_variables)
            #print("tower_grads",tower_grads)
            aver_loss_op = tf.reduce_mean(tower_losses)
            tf.summary.scalar('loss', aver_loss_op)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads), global_step=global_step)

            all_y = tf.reshape(tf.stack(tower_y, 0), [-1,N_CLASSES])
            all_pred = tf.reshape(tf.stack(tower_preds, 0), [-1,N_CLASSES])
            
            correct_prediction = tf.equal(tf.argmax(all_pred, 1), tf.argmax(all_y, 1))
            correct_prediction_top3 = tf.nn.in_top_k(all_pred, tf.argmax(all_y, 1), 3)
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            evaluation_step_top3 = tf.reduce_mean(tf.cast(correct_prediction_top3, tf.float32))
            correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
            correct_num_top3 = tf.reduce_sum(tf.cast(correct_prediction_top3, tf.int32))
            tf.summary.scalar('accuracy', evaluation_step)
            tf.summary.scalar('accuracy_top3', evaluation_step_top3)
            '''
            correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
            tf.summary.scalar('accuracy', accuracy)
            '''
            print('reduce model on cpu done.')
            startepoch = tf.get_variable('startepoch', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
            sess.run(tf.global_variables_initializer())
            
            var_list = tf.trainable_variables()
            #print("trainable_variables",var_list)
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            saver = tf.train.Saver(var_list=var_list, max_to_keep=3)
            summary_op = tf.summary.merge_all()
            print('run train op...')
            load_fn = slim.assign_from_checkpoint_fn(pre_trained_weights, get_tuned_variables(), ignore_missing_vars=True)
            load_fn(sess)
            print("load finished")
            
            ckpt_dir = train_log_dir
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            ckpt = tf.train.latest_checkpoint(train_log_dir)
            if ckpt != None:
                saver.restore(sess,ckpt)
            else:
                print("Training from scratch.")
            start = sess.run(startepoch)
            print("Training starts from {} epochs.".format(start+1))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            val_summary_writer = tf.summary.FileWriter(test_log_dir, sess.graph)
            #variable_name = [v.name for v in tf.trainable_variables()]
            #print("trainable_variables",variable_name)
            payload_per_gpu = BATCH_SIZE/num_gpu
            try:
                for epoch in range(start,epochs):
                    epoch_start_time = time()
                    for step in range(0,each_epoch_step):
                        if coord.should_stop():
                            break
                        global_steps = sess.run(global_step)
                        
                        #training
                        tra_images,tra_labels = sess.run([training_images,training_labels])
                        inp_dict = {}
                        inp_dict[keep_prob] = 0.8
                        inp_dicts = feed_all_gpu(inp_dict, models, payload_per_gpu, tra_images, tra_labels)
                        _ = sess.run(apply_gradient_op, inp_dicts)
                        if (global_steps+1)==10 or (global_steps+1) % 1000 == 0 or (global_steps+1) == MAX_STEPS:
                            tra_images,tra_labels = sess.run([training_images,training_labels])
                            inp_dict = {}
                            inp_dict[keep_prob] = 0.8
                            inp_dicts = feed_all_gpu(inp_dict, models, payload_per_gpu, tra_images, tra_labels)
                            summary_str,train_accuracy,train_accuracy_top3,tra_loss = sess.run([summary_op,evaluation_step,evaluation_step_top3,aver_loss_op], inp_dicts)
                            tra_summary_writer.add_summary(summary_str, global_steps+1)
                            print("After %d training steps，train_loss = %.4f,train_accuracy = %.2f%%,train_accuracy_top3 = %.2f%%" % 
                                  (global_steps+1,tra_loss,train_accuracy*100,train_accuracy_top3*100))
                        
                        if (global_steps+1)==10 or (global_steps+1) % 1000 == 0 or (global_steps+1) == MAX_STEPS:
                            val_images,val_labels = sess.run([validation_images, validation_labels])
                            inp_dict = {}
                            inp_dict[keep_prob] = 0.8
                            inp_dicts = feed_all_gpu(inp_dict, models, payload_per_gpu, val_images,val_labels )
                            summary_str,val_accuracy,val_accuracy_top3,val_loss = sess.run([summary_op,evaluation_step,evaluation_step_top3,aver_loss_op], inp_dicts)
                            val_summary_writer.add_summary(summary_str, global_steps+1)
                            print("After %d training steps，val_loss = %.4f,val_accuracy = %.2f%%,val_accuracy_top3 = %.2f%%" % 
                                  (global_steps+1,val_loss,val_accuracy*100,val_accuracy_top3*100))
                    #test         
                    if (epoch+1) % 2 == 0 or (epoch+1) == epochs:
                        total_correct = 0
                        total_correct_top3 = 0
                        for each_bach in range(num_batch):
                            tes_images,tes_labels = sess.run([test_images, test_labels])
                            inp_dict = {}
                            inp_dict[keep_prob] = 1.0
                            inp_dicts = feed_all_gpu(inp_dict, models, payload_per_gpu, tes_images, tes_labels)
                            batch_correct,batch_correct_top_3= sess.run([correct_num,correct_num_top3], inp_dicts)

                            #print("batch_correct:",batch_correct)
                            total_correct += np.sum(batch_correct)
                            total_correct_top3 += np.sum(batch_correct_top_3)
                        print('"After %d epochs，Total testing samples: %d' %(epoch+1,num_sample))
                        print('Top_1 test average accuracy: %.2f%%' %(100*total_correct/num_sample))
                        print('Top_3 test average accuracy: %.2f%%' %(100*total_correct_top3/num_sample))
                    
                    if (epoch+1) % 4 == 0 or (epoch+1) == epochs:
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=epoch+1)
                        print("save checkpoint epoch: %d" % (epoch+1))
                    epoch_end_time =time()
                    print("Current %d epoch takes:%.2f" % (epoch+1,epoch_end_time-epoch_start_time))
                    epoch_start_time = epoch_end_time
                    sess.run(startepoch.assign(epoch+1))
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)
            duration = time() - startTime
            print("Total Train Takes:",duration)
            print('training is done.')

if __name__ == '__main__':
    tf.app.run()