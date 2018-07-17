from __future__ import absolute_import;
from __future__ import print_function;
from __future__ import division;
from tensorflow.examples.tutorials.mnist import input_data;
import tensorflow as tf;
import sys;
import os;
import numpy as np;
sys.path.append('../');
import net;
from .restore import assign_from_checkpoint_fn;
import time;

def train(netname,dumpdir,settings,mnist=input_data.read_data_sets('/data4T1/samhu/MNIST',one_hot=True)):
    print('train/valid/test',mnist.train.num_examples,'/',mnist.validation.num_examples,'/',mnist.test.num_examples);
    epoch_len = mnist.train.num_examples // settings['batch_size'];
    settings['epoch_len'] = epoch_len;
    with tf.device('/gpu:0'):
        netDict = net.getNet(netname,settings);
    save_path = dumpdir+os.sep+netname;
    #config session;
    config = tf.ConfigProto();
    config.gpu_options.allow_growth = True;
    config.allow_soft_placement = True;
    if not os.path.exists(save_path):
        os.makedirs(save_path);
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer());
        saver = tf.train.Saver();
        train_writer = tf.summary.FileWriter("%s/train"%(save_path),graph=sess.graph);
        valid_writer = tf.summary.FileWriter("%s/valid"%(save_path),graph=sess.graph);
        ckpt = tf.train.get_checkpoint_state('%s/'%save_path);
        max_step = 30000;
        if ckpt and ckpt.model_checkpoint_path:
            assign = assign_from_checkpoint_fn(ckpt.model_checkpoint_path,tf.global_variables(),True);
            assign(sess);
        try:
            tstart = time.time();
            for i in range(max_step):
                xs, ys = mnist.train.next_batch(settings['batch_size']);
                rxs = np.reshape(xs,(settings['batch_size'], settings['imgH'],settings['imgW'],settings['imgC']));
                feed = {netDict['x2D']:rxs,netDict['yGT']:ys,netDict['keeprate']:0.5};
                _, loss_value,summary,step = sess.run([netDict['opt'],netDict['loss'],netDict['train_sum'],netDict['step']],feed_dict=feed);
                train_writer.add_summary(summary,step);
                if i%1000 == 0:
                    xv,yv = mnist.validation.next_batch(settings['test_batch_size']);
                    rxv = np.reshape(xv,(settings['test_batch_size'],settings['imgH'],settings['imgW'],settings['imgC']));
                    valid_feed = {netDict['x2D']:rxv,netDict['yGT']:yv,netDict['keeprate']:1.0};
                    acc,summary,step = sess.run([netDict['acc'],netDict['valid_sum'],netDict['step']],feed_dict=valid_feed);
                    valid_writer.add_summary(summary,step);
                    saver.save(sess, save_path+os.sep+'model.cpkt',global_step=step);
                    print("After %d training step(s), accuracy on validation is %f." % (step,acc));
                if (i%epoch_len == 0) or (i == max_step - 1):
                    acc_sum = 0.0;
                    acc_cnt = 0;
                    for itst in range(mnist.test.num_examples//settings['test_batch_size']):
                        xt,yt = mnist.test.next_batch(settings['test_batch_size'])
                        rxt = np.reshape(xt,(settings['test_batch_size'],settings['imgH'],settings['imgW'],settings['imgC']));
                        test_feed = {netDict['x2D']:rxt,netDict['yGT']:yt,netDict['keeprate']:1.0};
                        acc_sum += sess.run(netDict['acc'],feed_dict=test_feed);
                        acc_cnt += 1;
                    print("After %d training step(s), accuracy on test is %f." % (step, acc_sum/float(acc_cnt)));
                if (i == max_step - 1):
                    print("Time used %f s"%(time.time()-tstart));
        except Exception,e:
            print(e);
        else:
            print('done mnist.train');
    
    
def infer(netname,dumpdir,settings,mnist=input_data.read_data_sets('/data4T1/samhu/MNIST',one_hot=True)):
    print('done mnist.infer');
    