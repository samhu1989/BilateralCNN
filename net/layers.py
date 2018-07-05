import tensorflow as tf;
from tensorflow.contrib import layers;

def conv2d(in2D,size,reg=layers.l2_regularizer(1e-5),init=tf.truncated_normal_initializer(stddev=0.1),stride=[1,1,1,1],pad='SAME',name='conv'):
    assert size[2]==int(in2D.shape[3]);
    with tf.variable_scope(name) as scope:
        try:
            w = tf.get_variable('w',size,
                                initializer = init,
                                regularizer = reg
                               );
            b = tf.get_variable('b',size[-1],
                                initializer = tf.constant_initializer(0.0),
                                regularizer = reg
                               );
        except ValueError:
            scope.reuse();
            w = tf.get_variable('w');
            b = tf.get_variable('b');
        conv = tf.nn.conv2d(in2D,w,strides=stride,padding=pad);
        o2D = tf.nn.relu(tf.nn.bias_add(conv,b));
    return o2D;

def fc(in1D,odim,keeprate=1.0,reg=layers.l2_regularizer(1e-4),init=tf.truncated_normal_initializer(stddev=0.1),name='fc'):
    indim = int(in1D.shape[1])
    with tf.variable_scope(name) as scope:
        try:
            w = tf.get_variable('w',[indim,odim],
                                initializer=init,
                                regularizer=reg
                               );
            b = tf.get_variable('b',[odim],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=reg
                               );
        except ValueError:
            scope.reuse();
            w = tf.get_variable('w');
            b = tf.get_variable('b');
        fc = tf.nn.relu(tf.matmul(in1D,w) + b);
        fc = tf.nn.dropout(fc,keeprate);
    return fc;