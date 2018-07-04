from __future__ import absolute_import;
from __future__ import print_function;
from __future__ import division;
from tensorflow.examples.tutorials.mnist import input_data;
import tensorflow as tf;
import sys;
sys.path.append('../');
import net;

def train(netname,dumpdir,settings,mnist=input_data.read_data_sets('/data4T1/samhu/MNIST')):
    settings['epoch_len'] = mnist.train.num_examples // settings['batch_size'];
    netDict = net.getNet('LeNet',settings);
    print('done mnist.train');
    
    
def infer(netname,dumpdir,settings,mnist=input_data.read_data_sets('/data4T1/samhu/MNIST')):
    print('done mnist.infer');
    