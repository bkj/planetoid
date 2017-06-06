"""

    test-location.py
    
    Testing location inference on Twitter
"""

import sys
import ujson as json
import cPickle
import numpy as np
import pandas as pd
from collections import namedtuple
from pprint import pprint

from trans_model import trans_model as model

# --
# Config

config = json.load(open('./config.json'))['transductive']
pprint(config)
config['use_feature'] = False

# --
# IO

paths = {
    'graph' : './data/twitter/twitter-graph',
    'locations' : '/home/bjohnson/projects/sm-network/data/twitter/user-locations.jl',
    'lookup' : './data/twitter/id_lookup.tsv'
}

lookup = pd.read_csv(paths['lookup'], sep='\t')

locs = pd.DataFrame(map(json.loads, open(paths['locations']).readlines()))
locs.user_id = locs.user_id.astype('int')

uids = set(lookup.twitter_a)
locs2 = filter(lambda x: int(x['user_id']) in uids, locs)


y = np.zeros()
x = np.zeros((y.shape[0], 1))
graph = cPickle.load(open(paths['graph']))

train_data = [x, y, graph]

# --
# Define model

margs = namedtuple('my_args', ' '.join(config.keys()))

m = model(margs(**config))
m.add_data(*train_data)
m.build()

# Pre-training
m.init_train(
    init_iter_label=config['init_train']['iter_label'], 
    init_iter_graph=config['init_train']['iter_graph']
)

# --
# Train model

iter_cnt, max_accu = 0, 0
while True:
    # perform a training step
    m.step_train(
        max_iter=config['step_train']['max_iter'],
        iter_graph=config['step_train']['iter_graph'],
        iter_inst=config['step_train']['iter_inst'],
        iter_label=config['step_train']['iter_label'],
    )
    
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
