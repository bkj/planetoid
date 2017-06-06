"""

    test.py
    
    Merging `test_trans.py` and `test_ind.py` 
"""

import sys
import json
import cPickle
import argparse
import numpy as np
from collections import namedtuple
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset', type=str, default='cora')
    parser.add_argument('--inductive', help='inductive?', action="store_true")
    parser.add_argument('--config', help='path to config.json', type=str, default='config.json')
    
    parser.add_argument('--learning_rate', help='learning rate for supervised loss', type=float)
    parser.add_argument('--embedding_size', help='embedding dimensions', type=int)
    parser.add_argument('--window_size', help='window size in random walk sequences', type=int)
    parser.add_argument('--path_size', help='length of random walk sequences', type=int)
    parser.add_argument('--batch_size', help='batch size for supervised loss', type=int)
    parser.add_argument('--g_batch_size', help='batch size for graph context loss', type=int)
    parser.add_argument('--g_sample_size', help='batch size for label context loss', type=int)
    parser.add_argument('--neg_samp', help='negative sampling rate; zero means using softmax', type=int)
    parser.add_argument('--g_learning_rate', help='learning rate for unsupervised loss', type=float)
    parser.add_argument('--model_file', help='filename for saving models', type=str)
    parser.add_argument('--use_feature', help='whether use input features', type=bool)
    parser.add_argument('--update_emb', help='whether update embedding when optimizing supervised loss', type=bool)
    parser.add_argument('--layer_loss', help='whether incur loss on hidden layers', type=bool)
    return parser.parse_args()

def comp_accu(tpy, ty):
    pred = np.argmax(tpy, axis=1)
    act = np.argmax(ty, axis=1)
    return (pred == act).sum() * 1.0 / tpy.shape[0]

args = parse_args()

# --
# Config

config = json.load(open(args.config))

if args.inductive:
    from ind_model import ind_model as model
    config = config['inductive']
else:
    from trans_model import trans_model as model
    config = config['transductive']

# Add command line arguments
for k,v in vars(args).iteritems():
    if v:
        print >> sys.stderr, "overriding default %s" % k
        config[k] = v

pprint(config)

# --
# IO

objs = []
for nm in ['x', 'y', 'tx', 'ty', 'allx', 'graph']:
    try:
        fname = "data/{}.{}.{}".format('ind' if args.inductive else 'trans', args.dataset, nm)
        obj = cPickle.load(open(fname))
        objs.append(obj)
    except:
        print >> sys.stderr, 'test: could not load %s' % nm
        objs.append(None)

x, y, tx, ty, allx, graph = tuple(objs)

train_data = [x, y, allx, graph] if args.inductive else [x, y, graph]

# --
# Define model

margs = namedtuple('my_args', ' '.join(config.keys()))

from trans_model import trans_model
m = trans_model(margs(**config))
m.add_data(*train_data)
m.build()

# Pre-training
for i in range(config['init_train']['iter_label']):
    gx, gy = next(m.label_generator)
    loss = m.g_fn(gx, gy)
    sys.stdout.write('\r init_train: label %d %f' % (i, loss))
    sys.stdout.flush()


for i in range(config['init_train']['iter_graph']):
    gx, gy = next(m.graph_generator)
    loss = m.g_fn(gx, gy)
    sys.stdout.write('\r init_train: graph %d %f' % (i, loss))
    sys.stdout.flush()


# --
# Train model

iter_cnt, max_accu = 0, 0
while True:
    # perform a training step
    
    if random.random() < config['step_train']['iter_graph']:
        gx, gy = next(m.graph_generator)
        m.g_fn(gx, gy)
    
    if random.random() < config['step_train']['iter_inst']:
        x, y, index = next(m.inst_generator)
        m.train_fn(x, y, index)
    
    if random.random() < config['step_train']['iter_label']:
        gx, gy = next(m.label_generator)
        m.g_fn(gx, gy)
    
    # predict the dev set
    tpy = m.predict(tx)
    
    # compute the accuracy on the dev set
    accu = comp_accu(tpy, ty)
    
    sys.stdout.write("\r step_train\t%d\t%f\t%f" % (iter_cnt, accu, max_accu))
    sys.stdout.flush()
    
    iter_cnt += 1
    
    if accu > max_accu:
        # store the model if better result is obtained
        m.store_params()
        max_accu = max(max_accu, accu)
