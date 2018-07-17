# -*- coding: utf-8 -*-
import tensorflow as tf;
import os;
from tensorflow.python.framework import ops
path = os.path.dirname(os.path.realpath(__file__));
group_module=tf.load_op_library(path+'/group.so');
wgroup_module=tf.load_op_library(path+'/wgroup.so');

def knn(xyz,k):
	'''
Computes the distance of k nearest neighbors inside a point clouds
output: dist:  (batch_size,#point,k)   nearest k neighbor dist inside point set 
output: idx:  (batch_size,#point,k)   nearest k neighbor index inside point set
	'''
	return group_module.knn(xyz,k);
    
def wknn(f,k,dh,dw):
	'''
Computes the distance of k nearest neighbors inside a point clouds
output: dist:  (batch_size,h,w,k)   nearest k neighbor dist inside point set 
output: idx:  (batch_size,h,w,k,3)   nearest k neighbor index inside point set
	'''
	return wgroup_module.win_knn(f,k,dh,dw);
    
def gpucpu():
    xyz=np.random.randn(1,4*4,4).astype('float32');
    with tf.Session('') as sess:
        with tf.device('/cpu:0'):
            ixyzcpu=tf.Variable(xyz)
            distcpu,idxcpu=knn(ixyzcpu,4);
            print idxcpu.shape;
        sess.run(tf.global_variables_initializer())
        t0=time.time();
        dvalcpu,valcpu = sess.run([distcpu,idxcpu]);
        cputime = time.time()-t0;
        with tf.device('/gpu:0'):
            ixyzgpu=tf.Variable(xyz)
            distgpu,idxgpu=knn(ixyzgpu,4);
            print idxgpu.shape;
        sess.run(tf.global_variables_initializer())
        t0=time.time();
        dvalgpu,valgpu = sess.run([distgpu,idxgpu]);
        gputime = time.time()-t0;
    print "cputime",cputime;
    print "gputime",gputime;
    print "xyz:",xyz.shape;
    #print xyz
    print "valgpu:",valgpu.shape;
    #print valgpu;
    print "valcpu:",valcpu.shape;
    #print valcpu;
    print "(valgpu==valcpu) is ",(valgpu==valcpu).all();
    if np.size(valgpu) <= 128:
        print "output:";
        print "gpu:",valgpu;
        print "gpu dist:",dvalgpu;
        print "cpu:",valcpu;
        print "cpu dist:",dvalcpu;
    else:
        print "output size:",np.size(valgpu);
    if xyz.shape[0] == 1 and xyz.shape[2] < 10:
        distmat = np.sum(np.square(np.reshape(xyz,(xyz.shape[1],1,xyz.shape[2])) - np.reshape(xyz,(1,xyz.shape[1],xyz.shape[2]))),axis=2);
        print "distmat:",distmat;
    if not (valgpu==valcpu).all():
        itemindex = np.where( valgpu != valcpu )
        print itemindex;
        print valgpu[itemindex];
        print valcpu[itemindex];
        print dvalcpu[itemindex[:-1]];
        print dvalgpu[itemindex[:-1]];
        
def gputime():
    xyz=2.0*np.random.randn(1,10*10,4).astype('float32');
    with tf.Session('') as sess:
        with tf.device('/gpu:0'):
            ixyzgpu=tf.Variable(xyz)
            distgpu,idxgpu=knn(ixyzgpu,9,3,3);
            print idxgpu.shape;
        sess.run(tf.global_variables_initializer())
        t0=time.time();
        for i in xrange(100):
            dvalgpu,valgpu = sess.run([distgpu,idxgpu]);
        gputime = time.time()-t0;
    print "gputime",gputime;
    
def wknntest():
    f = 2.0*np.random.randn(1,4,4,4).astype('float32');
    with tf.Session('') as sess:
        with tf.device('/gpu:0'):
            ifgpu=tf.Variable(f);
            distgpu,idxgpu=wknn(ifgpu,4,2,2);
            print idxgpu.shape;
        with tf.device('/cpu:0'):
            ifcpu=tf.Variable(f);
            distcpu,idxcpu=wknn(ifcpu,4,2,2);
            print idxcpu.shape;
        sess.run(tf.global_variables_initializer());
        t0=time.time();
        dvalgpu,valgpu = sess.run([distgpu,idxgpu]);
        gputime = time.time()-t0;
        t0=time.time();
        dvalcpu,valcpu = sess.run([distcpu,idxcpu]);
        cputime = time.time()-t0;
        print f;
        print "cputime:", cputime;
        print "gputime:", gputime;
        if not (valgpu==valcpu).all():
            print "conflict result"
        print "valgpu:";
        print valgpu;
        print "dvalgpu";
        print dvalgpu;
        print "valcpu:";
        print valcpu;
        print "dvalcpu";
        print dvalcpu;
        
def wknntimegpu():
    f = 2.0*np.random.randn(32,256,256,4).astype('float32');
    with tf.Session('') as sess:
        with tf.device('/gpu:0'):
            ifgpu=tf.Variable(f);
            distgpu,idxgpu=wknn(ifgpu,16,7,7);
            print idxgpu.shape;
        sess.run(tf.global_variables_initializer());
        t0=time.time();
        dvalgpu,valgpu = sess.run([distgpu,idxgpu]);
        gputime = time.time()-t0;
        print "gputime:", gputime;
        
def wknntimecpu():
    f = 2.0*np.random.randn(32,256,256,4).astype('float32');
    with tf.Session('') as sess:
        with tf.device('/cpu:0'):
            ifcpu=tf.Variable(f);
            distcpu,idxcpu=wknn(ifcpu,9,3,3);
            print idxcpu.shape;
        sess.run(tf.global_variables_initializer());
        t0=time.time();
        dvalcpu,valcpu = sess.run([distcpu,idxcpu]);
        cputime = time.time()-t0;
        print "cputime:", cputime;

if __name__=='__main__':
    import numpy as np;
    import random;
    import time;
    os.environ['CUDA_VISIBLE_DEVICES']='0';
    wknntimegpu();


