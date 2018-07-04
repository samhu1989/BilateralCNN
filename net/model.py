from __future__ import absolute_import;
from __future__ import print_function;
import os;
import sys;
#dynamic import the network modules to local namespace
from importlib import import_module;
for pn,dns,fns in os.walk(os.path.dirname(__file__)):
    mn = pn.split(os.sep)[-1];
    for fn in fns:
        if fn.startswith('m_') and fn.endswith('.py'):
            submn = fn.split('.')[0];
            m = import_module('%s.%s'%(mn,submn));
            for name in dir(m):
                if not name in locals().keys():
                    locals()[name] = m.__getattribute__(name);
                    
def getNet(name,settings={}):
    return eval(name)(settings);