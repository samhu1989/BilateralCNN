import tensorflow as tf;
from tensorflow.contrib import layers;
import numpy as np;
from .group import wknn;
from .layers import fc;

#this version is too slow
def bconv2d_v2(in2D,size,k,reg=layers.l2_regularizer(1e-5),init=tf.truncated_normal_initializer(stddev=0.1),name='conv'):
    #k should be odd
    k = (k//2)*2+1;
    assert size[2]==int(in2D.shape[3]);
    w = int(in2D.shape[1]);
    h = int(in2D.shape[2]);
    d = int(in2D.shape[3]);
    od = int(size[3]);
    xcoordv = np.reshape(np.stack(np.meshgrid(np.arange(w),np.arange(h)),axis=2),[1,w,h,2]);
    with tf.variable_scope(name) as scope:
        try:
            sf = tf.get_variable('sf',[1,1,d,2],initializer = init,regularizer = reg);
            bs = tf.get_variable('bs',[1,1,1,4],initializer = tf.constant_initializer(1.0),regularizer = reg);
            bb = tf.get_variable('bb',[1,1,1,4],initializer = tf.constant_initializer(0.0),regularizer = reg);
            b = tf.get_variable('b',[od],initializer = tf.constant_initializer(0.0),regularizer = reg);
        except ValueError:
            scope.reuse();
            sf = tf.get_variable('sf');
            bs = tf.get_variable('bs');
            bb = tf.get_variable('bb');
            b = tf.get_variable('b');
        fcoord = tf.nn.conv2d(in2D,sf,strides=[1,1,1,1],padding="VALID");
        xcoord = tf.constant(xcoordv,dtype=tf.float32);
        xcoord = tf.tile(xcoord,[tf.shape(in2D)[0],1,1,1]);
        coord = tf.concat([fcoord,xcoord],3);
        coordm,coordvar = tf.nn.moments(coord,[a for a in range(1,len(coord.shape)-1)],keep_dims=True,name='moments');
        coord = tf.nn.batch_normalization(coord,coordm,coordvar,bs,bb,1e-5);
        _,kidx = wknn(coord,k,(size[0] - 1)//2,(size[1]-1)//2);
        localcoord = tf.gather_nd(coord,kidx);
        localcoord -= tf.reshape(coord,[tf.shape(in2D)[0],h,w,1,4]);
        localcoord = tf.reshape(localcoord,[-1,4]);
        W = fc(localcoord,k-1,reg=reg,init=init,name=name+'_fc1');
        W = fc(W,d*od,activate='linear',reg=reg,init=init,name=name+'_fc2');
        W = tf.reshape(W,[tf.shape(in2D)[0],h,w,k,d,od]);
        fkn = tf.reshape(tf.gather_nd(in2D,kidx,name='fkn'),[-1,h,w,k,d,1]);
        fkn *= W;
        conv = tf.reduce_sum(fkn,axis=[3,4]);
        conv = tf.reshape(conv,[tf.shape(in2D)[0],h,w,od]);
        o2D = tf.nn.relu(tf.nn.bias_add(conv,b));
    return o2D;

def bconv2d_v1(in2D,size,k,reg=layers.l2_regularizer(1e-5),init=tf.truncated_normal_initializer(stddev=0.1),name='conv'):
    #k should be odd
    k = (k//2)*2+1;
    assert size[2]==int(in2D.shape[3]);
    w = int(in2D.shape[1]);
    h = int(in2D.shape[2]);
    d = int(in2D.shape[3]);
    od = int(size[3]);
    xcoordv = np.reshape(np.stack(np.meshgrid(np.arange(w),np.arange(h)),axis=2),[1,w,h,2]);
    with tf.variable_scope(name) as scope:
        try:
            sf = tf.get_variable('sf',[1,1,d,2],initializer = init,regularizer = reg);
            bs = tf.get_variable('bs',[1,1,1,4],initializer = tf.constant_initializer(1.0),regularizer = reg);
            bb = tf.get_variable('bb',[1,1,1,4],initializer = tf.constant_initializer(0.0),regularizer = reg);
            b = tf.get_variable('b',[od],initializer = tf.constant_initializer(0.0),regularizer = reg);
            W = tf.get_variable('W',[1,k,d,od],initializer = init,regularizer = reg);
        except ValueError:
            scope.reuse();
            sf = tf.get_variable('sf');
            bs = tf.get_variable('bs');
            bb = tf.get_variable('bb');
            b = tf.get_variable('b');
            W = tf.get_variable('W');
        fcoord = tf.nn.conv2d(in2D,sf,strides=[1,1,1,1],padding="VALID");
        xcoord = tf.constant(xcoordv,dtype=tf.float32);
        xcoord = tf.tile(xcoord,[tf.shape(in2D)[0],1,1,1]);
        coord = tf.concat([fcoord,xcoord],3);
        coordm, coordvar = tf.nn.moments(coord,[a for a in range(1,len(coord.shape)-1)],keep_dims=True,name='moments');
        coord = tf.nn.batch_normalization(coord,coordm,coordvar,bs,bb,1e-5);
        _,kidx = wknn(coord,k,(size[0]-1)//2,(size[1]-1)//2);
        fkn = tf.reshape(tf.gather_nd(in2D,kidx,name='fkn'),[-1,h*w,k,size[2]]);
        xkn = tf.reshape(tf.gather_nd(coord,kidx,name='xkn'),[-1,h*w,k,4]);
        xkn -= tf.reshape(coord,[-1,h*w,1,4]);
        wkn = tf.nn.softmax(tf.negative(tf.reduce_sum(tf.square(xkn),axis=3,keep_dims=True)));
        fkn *= wkn;
        conv = tf.nn.conv2d(fkn,W,strides=[1,1,1,1],padding="VALID");
        conv = tf.reshape(conv,[-1,h,w,od]);
        o2D = tf.nn.relu(tf.nn.bias_add(conv,b));
    return o2D;