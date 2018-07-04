import tensorflow as tf;
from .layers import conv2d;
from .layers import fc;

def LeNet(settings={}):
    net_dict={};
    if 'batch_size' in setttings.keys():
        batch_size = settings['batch_size'];
    else:
        batch_size = 32;
    if 'imgH' in setttings.keys():
        h = settings['imgH'];
    else:
        h = 28;
    if 'imgW' in setttings.keys():
        w = settings['imgW'];
    else:
        w = 28;
    if 'yDim' in setttings.keys():
        w = settings['yDim'];
    else:
        yDim = 10;
    x2D = tf.placeholder(tf.float32,[
                                   None,
                                   h,
                                   w,
                                   c
                                   ],name='x2D');
    net_dict['x2D'] = x2D;
    yGT = tf.placeholder(tf.float32, [None,yDim],name='yGT');
    net_dict['yGT'] = yGT;
    conv1 = conv2d(x2D,size=[5,5,1,32],name='conv1');
    net_dict['conv1'] = conv1;
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');
    net_dict['pool1'] = pool1;
    conv2 = conv2d(pool1,size=[5,5,32,64],name='conv2');
    net_dict['conv2'] = conv2;
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');
    net_dict['pool2'] = pool2;
    pool2_shape = pool2.get_shape().as_list();
    pool2_dim = pool2_shape[1] * pool2_shape[2] * pool2_shape[3];
    pool2_reshaped = tf.reshape(pool2,[-1,pool2_dim]);
    fc1 = fc(pool2_reshaped,512,istrain=settings['istrain'],name='fc1');
    net_dict['fc1']=fc1;
    y = fc(fc1,10,istrain=settings['istrain'],name='fc2');
    net_dict['y']=y;
    
    gstep = tf.Variable(0,trainable=False);
    net_dict['step']=gstep;
    
    variable_average = tf.train.ExponentialMovingAverage(0.99,gstep);
    variable_average_op = variable_average.apply(tf.trainable_variables());
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y);
    cross_entropy_mean = tf.reduce_mean(cross_entropy);

    loss = cross_entropy_mean + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES));

    learning_rate = tf.train.exponential_decay(0.8,global_step=gstep, decay_steps=mnist.train.num_examples / batch_size,decay_rate=0.99);
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=gstep)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train');
    net_dict['opt'] = train_op;
    return net_dict;