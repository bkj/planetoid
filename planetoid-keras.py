import theano
import theano.tensor as T
import lasagne

from keras import backend as K
from keras.models import Model
from keras import optimizers
from keras.layers import Input, Embedding, Dense, Flatten

from lasagne.layers import get_output as lasagne_get_output
from lasagne.layers import get_all_params as lasagne_get_params

# --
# Params

num_ver = 2709
emb_dim = 50
neg_samp = 0
assert(neg_samp == 0)
lr = 0.01

# --

m = trans_model(margs(**config))
m.add_data(*train_data)
m.build()

# --
# Lasagne model

edge_ = T.imatrix('edge')
input_edge = lasagne.layers.InputLayer(shape=(None, 2), input_var=edge_)

source_layer = lasagne.layers.SliceLayer(input_edge, indices=0, axis=1)
target_layer = lasagne.layers.SliceLayer(input_edge, indices=1, axis=1)

source_emb = lasagne.layers.EmbeddingLayer(source_layer, input_size=num_ver, output_size=emb_dim)
l_gy = lasagne.layers.DenseLayer(source_emb, num_ver, nonlinearity=lasagne.nonlinearities.softmax)

g_loss = lasagne.objectives.categorical_crossentropy(lasagne_get_output(l_gy), lasagne_get_output(target_layer)).sum()
g_params  = lasagne_get_params(l_gy, trainable=True)
g_updates = lasagne.updates.sgd(g_loss, g_params, learning_rate=lr)

g_fn = theano.function([edge_], g_loss, updates=g_updates, on_unused_input='ignore')

# --
# Keras model

inp_source = Input(shape=(1,))
source_emb = Embedding(input_dim=num_ver, output_dim=emb_dim)(inp_source)
source_emb = Flatten()(source_emb)
source_den = Dense(num_ver, activation='softmax')(source_emb)

model = Model(input=inp_source, output=source_den)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

# --
# Train side-by-side

K.set_value(model.optimizer.lr, lr * 100)
for i in range(config['init_train']['iter_label']):
    gx, gy = next(m.label_generator)
    orig_loss = g_fn(gx)
    print orig_loss
    new_loss = model.train_on_batch(gx[:,0].reshape(-1, 1), gx[:,1].reshape(-1, 1))
    print new_loss
    print


K.set_value(model.optimizer.lr, lr * 130)
for i in range(config['init_train']['iter_graph']):
    gx, gy = next(m.graph_generator)
    orig_loss = g_fn(gx)
    print orig_loss
    _ = model.fit(gx[:,0].reshape(-1, 1), gx[:,1].reshape(-1, 1), nb_epoch=1, batch_size=100, verbose=False)
    new_loss = model.evaluate(gx[:,0].reshape(-1, 1), gx[:,1].reshape(-1, 1), verbose=False) * gx.shape[0]
    print new_loss
    print

