from tensorflow.examples.tutorials.mnist import input_data;
import tensorflow as tf;
from __future__ import absolute_import;
from __future__ import print_function;
import sys;
sys.path.append('../');
import net;


def train(netname,dumpdir,mnist=input_data.read_data_sets('/data4T1/samhu/MNIST')):
    settings={};
    settings['istrain'] = True;
    settings['batch_size'] = 32;
    netDict = net.getNet('LeNet',settings);
    print('training');
    
    
    
    
    
    
def infer(netname,dumpdir,mnist=input_data.read_data_sets('/data4T1/samhu/MNIST')):
    print('evaluate');
    