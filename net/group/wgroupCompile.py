# -*- coding: utf-8 -*-
import os;
import tensorflow as tf;
nvcc = "/usr/local/cuda-8.0/bin/nvcc";
cxx = "g++";
cudalib = "/usr/local/cuda-8.0/lib64/";
TF_INC = tf.sysconfig.get_include();
TF_LIB = tf.sysconfig.get_lib();

os.system(nvcc+" -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o ./wgroup.cu.o ./wgroup.cu -I "+TF_INC+" -I "+TF_INC+"/external/nsync/public"+" -L"+TF_LIB+" -ltensorflow_framework"+" -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2");
os.system(cxx+" -std=c++11 ./wgroup.cpp ./wgroup.cu.o -o ./wgroup.so -shared -fPIC -I "+TF_INC+" -I "+TF_INC+"/external/nsync/public"+" -lcudart -L "+cudalib+" -L"+TF_LIB+" -ltensorflow_framework"+" -O2 -D_GLIBCXX_USE_CXX11_ABI=0");

