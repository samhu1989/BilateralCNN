import tensorflow as tf;
from .bconv import bconv2d_v1 as bconv2d;
from .layers import conv2d;
from .layers import fc;

def LeNetB(settings={}):
    net_dict={};
    epoch_len = settings['epoch_len'];
    if 'batch_size' in settings.keys():
        batch_size = settings['batch_size'];
    else:
        batch_size = 32;
    if 'imgH' in settings.keys():
        h = settings['imgH'];
    else:
        h = 28;
    if 'imgW' in settings.keys():
        w = settings['imgW'];
    else:
        w = 28;
    if 'imgC' in settings.keys():
        c = settings['imgC'];
    else:
        c = 1;
    if 'yDim' in settings.keys():
        yDim = settings['yDim'];
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
    dr = tf.placeholder(tf.float32,name='keeprate');
    net_dict['keeprate'] = dr;
    net_dict['train_sum'] = [];
    net_dict['valid_sum'] = [];
    conv1 = bconv2d(x2D,size=[11,11,1,32],k=23,name='conv1');
    #conv1 = conv2d(x2D,size=[5,5,1,32],name='conv1');
    net_dict['conv1'] = conv1;
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');
    net_dict['pool1'] = pool1;
    conv2 = bconv2d(pool1,size=[11,11,32,64],k=23,name='conv2');
    net_dict['conv2'] = conv2;
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');
    net_dict['pool2'] = pool2;
    pool2_shape = pool2.get_shape().as_list();
    pool2_dim = pool2_shape[1] * pool2_shape[2] * pool2_shape[3];
    pool2_reshaped = tf.reshape(pool2,[-1,pool2_dim]);
    fc1 = fc(pool2_reshaped,16,keeprate=dr,name='fc1');
    net_dict['fc1']=fc1;
    y = fc(fc1,10,keeprate=1.0,name='fc2');
    net_dict['y']=y;
    
    gstep = tf.Variable(0,trainable=False);
    net_dict['step']=gstep;
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(yGT, 1), logits=y);
    cross_entropy_mean = tf.reduce_mean(cross_entropy);
    summary = tf.summary.scalar('cross_entropy',cross_entropy_mean);
    net_dict['train_sum'].append(summary);
    net_dict['valid_sum'].append(summary);

    reg = 1e-4*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES));
    net_dict['reg'] = reg;
    summary = tf.summary.scalar('reg',reg);
    net_dict['train_sum'].append(summary);
    net_dict['valid_sum'].append(summary);
    
    loss = cross_entropy_mean + reg;
    net_dict['loss'] = loss;
    summary = tf.summary.scalar('loss',loss);
    net_dict['train_sum'].append(summary);
    net_dict['valid_sum'].append(summary);

    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=gstep)
        
    correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(yGT, 1));
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
    net_dict['acc'] = accuracy;
    summary = tf.summary.scalar('accuracy',accuracy);
    net_dict['train_sum'].append(summary);
    net_dict['valid_sum'].append(summary);
    
    net_dict['opt'] = train_op;
    net_dict['train_sum'] = tf.summary.merge(net_dict['train_sum']);
    net_dict['valid_sum'] = tf.summary.merge(net_dict['valid_sum']);
    print('got LeNetB');
    return net_dict;