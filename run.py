import tensorflow as tf;
import argparse;
import sys;
import util;

if __name__ == '__main__':
    parse = argparse.ArgumentParser();
    parse.add_argument("--data",type=str,default='mnist',help="data path");
    parse.add_argument("--net",type=str,default='LeNet',help="network name");
    parse.add_argument("--dump",type=int,default=100,help="dump path where the trained model and other information is saved");
    parse.add_argument("--cmd",type=str,default='train',help="command");
    flags,unparsed = parse.parse_known_args(sys.argv[1:]);
    settings={}
    if flags.cmd == 'train':
        util.train()

