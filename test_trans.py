
import cPickle
import argparse
import numpy as np
from scipy import sparse as sp

from trans_model import trans_model as model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',         help='dataset', type=str, default='citeseer')
    parser.add_argument('--learning_rate',   help='learning rate for supervised loss', type=float, default=0.1)
    parser.add_argument('--embedding_size',  help='embedding dimensions', type=int, default=50)
    parser.add_argument('--window_size',     help='window size in random walk sequences', type=int, default=3)
    parser.add_argument('--path_size',       help='length of random walk sequences', type=int, default=10)
    parser.add_argument('--batch_size',      help='batch size for supervised loss', type=int, default=200)
    parser.add_argument('--g_batch_size',    help='batch size for graph context loss', type=int, default=200)
    parser.add_argument('--g_sample_size',   help='batch size for label context loss', type=int, default=100)
    parser.add_argument('--neg_samp',        help='negative sampling rate; zero means using softmax', type=int, default=0)
    parser.add_argument('--g_learning_rate', help='learning rate for unsupervised loss', type=float, default=1e-2)
    parser.add_argument('--model_file',      help='filename for saving models', type=str, default='trans.model')
    parser.add_argument('--use_feature',     help='whether use input features', type=bool, default=True)
    parser.add_argument('--update_emb',      help='whether update embedding when optimizing supervised loss', type=bool, default=True)
    parser.add_argument('--layer_loss',      help='whether incur loss on hidden layers', type=bool, default=True)
    return parser.parse_args()

def comp_accu(tpy, ty):
    pred = np.argmax(tpy, axis=1)
    act = np.argmax(ty, axis=1)
    return (pred == act).sum() * 1.0 / tpy.shape[0]


args = parse_args()


# --
# IO

objs = []
for nm in ['x', 'y', 'tx', 'ty', 'graph']:
    fname = "data/trans.{}.{}".format(args.dataset, nm)
    obj = cPickle.load(open(fname))
    objs.append(obj)

x, y, tx, ty, graph = tuple(objs)

# --
# Define model

m = model(args)
m.add_data(x, y, graph)                                 # add data
m.build()                                               # build the model
m.init_train(init_iter_label=2000, init_iter_graph=70)  # pre-training

# --
# Train model

iter_cnt, max_accu = 0, 0
while True:
    # perform a training step
    m.step_train(max_iter=1, iter_graph=0, iter_inst=1, iter_label=0)

    # predict the dev set
    tpy = m.predict(tx)

    # compute the accuracy on the dev set
    accu = comp_accu(tpy, ty)

    print "Step : %d, %f, %f" % (iter_cnt, accu, max_accu)
    iter_cnt += 1

    if accu > max_accu:
        # store the model if better result is obtained
        m.store_params()
        max_accu = max(max_accu, accu)
