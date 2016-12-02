import sys
import copy
import random
import theano
import lasagne
import numpy as np

from lasagne.layers import get_output as lasagne_get_output
from lasagne.layers import get_all_params as lasagne_get_params

import theano.tensor as T
from theano.sparse import csr_matrix as icsr_matrix
from collections import defaultdict

import layers
from base_model import base_model

class trans_model(base_model):
    """ Planetoid-T """
    
    def add_data(self, x, y, graph):
        """
        add data to the model.
        
        x (scipy.sparse.csr_matrix) : feature vectors for training data.
        y (numpy.ndarray)           : one-hot label encoding for training data.
        graph (dict)                : the format is {index: list_of_neighbor_index}. Only supports binary graph.

        Let L and U be the number of training and dev instances.
        The training instances must be indexed from 0 to L - 1 with the same order in x and y.
        By default, our implementation assumes that the dev instances are indexed from L to L + U - 1, unless otherwise
        specified in self.predict.
        """
        self.x, self.y, self.graph = x, y, graph
        self.num_ver = max(self.graph.keys()) + 1
        
    def build(self):
        assert(self.x != None)
        assert(self.y != None)
        assert(self.graph != None)
        
        self.l = []
        
        # --
        # `g_fn`
        edge_  = T.imatrix('edge')
        input_edge = lasagne.layers.InputLayer(shape=(None, 2), input_var=edge_)
        
        source_layer = lasagne.layers.SliceLayer(input_edge, indices=0, axis=1)
        target_layer = lasagne.layers.SliceLayer(input_edge, indices=1, axis=1)
        
        # ?? Should these be tied?
        source_emb = lasagne.layers.EmbeddingLayer(source_layer, input_size=self.num_ver, output_size=self.embedding_size)
        target_emb = lasagne.layers.EmbeddingLayer(target_layer, input_size=self.num_ver, output_size=self.embedding_size)
        
        edge_label_ = T.vector('edge_label')
        if self.neg_samp == 0:
            # Predict neighbor
            l_gy   = layers.DenseLayer(source_emb, self.num_ver, nonlinearity=lasagne.nonlinearities.softmax)
            g_loss = lasagne.objectives.categorical_crossentropy(lasagne_get_output(l_gy), lasagne_get_output(target_layer)).sum()
        else:
            # Inner product w/ neighbor's embedding
            l_gy   = lasagne.layers.ElemwiseMergeLayer([source_emb, target_emb], T.mul)
            g_loss = -T.log(T.nnet.sigmoid(T.sum(lasagne_get_output(l_gy), axis=1) * edge_label_)).sum()
        
        g_params  = lasagne_get_params(l_gy, trainable=True)
        g_updates = lasagne.updates.sgd(g_loss, g_params, learning_rate=self.g_learning_rate)
        self.g_fn = theano.function([edge_, edge_label_], g_loss, updates=g_updates, on_unused_input='ignore')
        self.l.append(l_gy)
        
        # --
        # `train_fn` -- only linked to `g_fn because `W` is tied to `source_emb`
        node_      = T.ivector('node')
        input_node = lasagne.layers.InputLayer(shape=(None, ), input_var=node_)
        node_emb   = lasagne.layers.EmbeddingLayer(
            input_node, input_size=self.num_ver, output_size=self.embedding_size, W=source_emb.W)
        node_emb   = layers.DenseLayer(
            node_emb, self.y.shape[1], nonlinearity=lasagne.nonlinearities.softmax)
        
        # Features
        x_       = icsr_matrix('x', dtype='float32')
        input_x  = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=x_)
        hidden_x = layers.SparseLayer(input_x, self.y.shape[1], nonlinearity=lasagne.nonlinearities.softmax)
        
        # Compute loss
        y_ = T.imatrix('y')
        if self.use_feature:            
            # Concat graph + content features
            hidden = lasagne.layers.ConcatLayer([hidden_x, node_emb], axis=1)
            
            # Dense softmax should be close to `y`
            hidden = layers.DenseLayer(hidden, self.y.shape[1], nonlinearity=lasagne.nonlinearities.softmax)
            loss = lasagne.objectives.categorical_crossentropy(lasagne_get_output(hidden), y_).mean()
            
            if self.layer_loss:
                # Features should be projected near label
                loss += lasagne.objectives.categorical_crossentropy(lasagne_get_output(hidden_x), y_).mean()
                
                # Embedding should be near label
                loss += lasagne.objectives.categorical_crossentropy(lasagne_get_output(node_emb), y_).mean()
            
        else:
            # Dense softmax should be close to `y`
            hidden = node_emb
            loss = lasagne.objectives.categorical_crossentropy(lasagne_get_output(hidden), y_).mean()
        
        # !! I don't understand this bit
        if self.use_feature:
            params = [node_emb.W, node_emb.b, hidden_x.W, hidden_x.b, hidden.W, hidden.b]
        elif self.update_emb:
            params = lasagne_get_params(hidden)
        else:
            params = [hidden.W, hidden.b]
        
        updates       = lasagne.updates.sgd(loss, params, learning_rate=self.learning_rate)
        self.train_fn = theano.function([x_, y_, node_], loss, updates=updates, on_unused_input='ignore')
        self.test_fn  = theano.function([x_, node_], lasagne_get_output(hidden), on_unused_input='ignore')
        self.l.append(hidden)
    
    def gen_train_inst(self):
        """
            generator for batches for classification loss.
            
            feats get mapped close to labels
            
            !! Could replace with array_split
        """
        
        while True:
            # Shuffle indices
            ind = np.array(np.random.permutation(self.x.shape[0]), dtype=np.int32)
            i = 0
            while i < ind.shape[0]:
                j = min(ind.shape[0], i + self.batch_size)
                
                # Yield x, y, ind
                yield self.x[ind[i:j]], self.y[ind[i:j]], ind[i:j]
                i = j
    
    def gen_label_graph(self):
        """
            generator for batches for label context loss.
            
            nodes are close to other nodes w/ same label
        """
        labels, label2inst, not_label = [], defaultdict(list), defaultdict(list)
        
        for i in xrange(self.x.shape[0]):
            flag = False
            for j in xrange(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)
        
        # !! Equiv to
        # labels     = y.argmax(axis=1)
        # label2inst = dict([(i, list(np.where(labels_ == i)[0])) for i in range(y.shape[1])])
        # not_label  = dict([(i, list(np.where(labels_ != i)[0])) for i in range(y.shape[1])])
        # labels     = list(labels_)
        
        while True:
            g, gy = [], []
            for _ in range(self.g_sample_size):
                
                # Sample random obs + label
                x1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[x1]
                
                # If only obs for the label, skip
                if len(label2inst) == 1:
                    continue
                
                # Randomly pick another instance w/ same label, and record as positive
                x2 = random.choice(label2inst[label])
                g.append([x1, x2])
                gy.append(1.0)
                
                # Add negative samples
                for _ in range(self.neg_samp):
                    g.append([x1, random.choice(not_label[label])])
                    gy.append(-1.0)
            
            yield np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)

    def gen_graph(self):
        """
            generator for batches for graph context loss.
            
            nodes are close to nodes near them in the graph
        """
        
        while True:
            # Permute vertices
            ind = np.random.permutation(self.num_ver)
            i = 0
            while i < ind.shape[0]:
                g, gy = [], []
                j = min(ind.shape[0], i + self.g_batch_size)
                
                # For each node
                for k in ind[i:j]:
                    # If not linked, skip
                    if len(self.graph[k]) == 0:
                        continue
                    
                    # Random walk from k
                    path = [k]
                    for _ in range(self.path_size):
                        path.append(random.choice(self.graph[path[-1]]))
                    
                    for l in range(len(path)):
                        for m in range(l - self.window_size, l + self.window_size + 1):
                            # If OOB, skip
                            if m < 0 or m >= len(path):
                                continue
                            
                            # Else, append record of being in same window
                            g.append([path[l], path[m]])
                            gy.append(1.0)
                            
                            # Add negative examples
                            for _ in range(self.neg_samp):
                                g.append([path[l], random.randint(0, num_ver - 1)])
                                gy.append(-1.0)
                
                yield np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)
                i = j

    def init_train(self, init_iter_label, init_iter_graph):
        """
        pre-training of graph embeddings.
        
        init_iter_label (int): # iterations for optimizing label context loss.
        init_iter_graph (int): # iterations for optimizing graph context loss.
        """
        for i in range(init_iter_label):
            gx, gy = next(self.label_generator)
            loss = self.g_fn(gx, gy)
            sys.stdout.write('\r init_train: label %d %f' % (i, loss))
            sys.stdout.flush()
        print
        
        for i in range(init_iter_graph):
            gx, gy = next(self.graph_generator)
            loss = self.g_fn(gx, gy)
            sys.stdout.write('\r init_train: graph %d %f' % (i, loss))
            sys.stdout.flush()
        print

    def step_train(self, max_iter, iter_graph, iter_inst, iter_label):
        """
        a training step. Iteratively sample batches for three loss functions.
        
        max_iter   (int) : # iterations for the current training step.
        iter_graph (int) : # iterations for optimizing the graph context loss.
        iter_inst  (int) : # iterations for optimizing the classification loss.
        iter_label (int) : # iterations for optimizing the label context loss.
        """
        for _ in range(max_iter):
            for _ in range(self.comp_iter(iter_graph)):
                gx, gy = next(self.graph_generator)
                self.g_fn(gx, gy)
            
            for _ in range(self.comp_iter(iter_inst)):
                x, y, index = next(self.inst_generator)
                self.train_fn(x, y, index)
            
            for _ in range(self.comp_iter(iter_label)):
                gx, gy = next(self.label_generator)
                self.g_fn(gx, gy)

    def predict(self, tx, index=None):
        """
        predict the dev or test instances.
        
        tx (scipy.sparse.csr_matrix): feature vectors for dev instances.
        index (numpy.ndarray): indices for dev instances in the graph. By default, we use the indices from L to L + U - 1.

        returns (numpy.ndarray, #instances * #classes): classification probabilities for dev instances.
        """
        if index is None:
            index = np.arange(self.x.shape[0], self.x.shape[0] + tx.shape[0], dtype=np.int32)
        
        return self.test_fn(tx, index)
