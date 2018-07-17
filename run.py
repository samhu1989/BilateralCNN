import tensorflow as tf;
import argparse;
import sys;
import util;
import os;

if __name__ == '__main__':
    parse = argparse.ArgumentParser();
    parse.add_argument("--data",type=str,default='mnist',help="data path");
    parse.add_argument("--net",type=str,default='LeNet',help="network name");
    parse.add_argument("--dump",type=str,help="dump path where the trained model and other information is saved");
    parse.add_argument("--gpu",type=str,default='',help="dump path where the trained model and other information is saved");
    parse.add_argument("--cmd",type=str,default='train',help="command");
    parse.add_argument("--batch_size",type=int,default=64,help="set batch size");
    parse.add_argument("--test_batch_size",type=int,default=100,help="set test batch size");
    parse.add_argument("--img_width",type=int,default=28,help="set image width");
    parse.add_argument("--img_height",type=int,default=28,help="set image height");
    parse.add_argument("--img_channel",type=int,default=1,help="set image channel");
    flags = parse.parse_args();
    settings={};
    settings['batch_size'] = flags.batch_size;
    settings['test_batch_size'] = flags.test_batch_size;
    settings['imgW'] = flags.img_width;
    settings['imgH'] = flags.img_height;
    settings['imgC'] = flags.img_channel;
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu;
    try:
        if flags.cmd == 'train':
            if flags.data == 'mnist':
                util.mnist.train(flags.net,flags.dump,settings);
        elif flags.cmd == 'infer':
            if flags.data == 'mnist':
                util.mnist.infer(flags.net,flags.dump,settings);
    except Exception,e:
        print(e);
    else:
        print('done');
                
            

