from __future__ import absolute_import, division, print_function
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, layers
from tensorflow.keras.callbacks import Callback
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import energyflow as ef
#from energyflow.archs.efn import PFN
#from energyflow.archs.dnn import DNN
#from energyflow.archs import DNN

#from energyflow.datasets import qg_jets
from energyflow.datasets import ttag_jets
from energyflow.utils import data_split, remap_pids, to_categorical

tf.experimental.numpy.experimental_enable_numpy_behavior()
from tensorflow.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import time
import os

########## Load True Data
true_data_num = 50000 #1000000
train_ratio, val_ratio, test_ratio = 0.75, 0.125, 0.125
train_true, val_true, test_true = int(true_data_num*train_ratio), int(true_data_num*val_ratio), int(true_data_num*test_ratio)
use_pids = False

print('Loading true dataset ...')
X_true, y_true = ttag_jets.load(train_true + val_true + test_true, generator='pythia')
# X_true, y_true = ttag_jets.load(train_true + val_true + test_true, generator='herwig', cache_dir='~/.energyflow/herwig')
n_true_pad = 200 - X_true.shape[1]
X_true = np.lib.pad(X_true, ((0,0), (0,n_true_pad), (0,0)), mode='constant', constant_values=0)
print('Dataset loaded!')

# convert labels to categorical
Y_true = to_categorical(y_true, num_classes=2)
print('Loaded Top and QCD jets')

# preprocess by centering jets, but not normalizing pts
for x in X_true:
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    #x[mask,0] /= x[:,0].sum()

# handle particle id channel
if use_pids:
    remap_pids(X_true, pid_i=3)
else:
    X_true = X_true[:,:,:3]
print('Finished preprocessing')

# do train/val/test split
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = data_split(X_true, Y_true, val=val_true, test=test_true, shuffle=False)
print('Done pythia train/val/test split')

X_true_flat = X_true.reshape(-1, X_true.shape[1]*X_true.shape[2])
X_train_flat = X_train.reshape(-1,X_train.shape[1]*X_train.shape[2])
X_val_flat = X_val.reshape(-1,X_val.shape[1]*X_val.shape[2])
X_test_flat = X_test.reshape(-1,X_test.shape[1]*X_test.shape[2])

########## Parameters
# dataset
m         = 200       # particles per jet, note above has a padding step, be sure to change that also
n_load    = true_data_num    # total true jets available
n_synth   = 5000     # how many synthetic jets per batch
latent_dim= 64       # noise size

patience_num = 20

# dnn 
nLayers = 2
layerSize = 100
dense_sizes = (layerSize,)*nLayers # student architecture
activations = ('relu',)*nLayers
drop_rates = (0.,)*nLayers
dnn_epochs = 200 # epochs per student, but doing early stopping now
inner_lr = 1e-3
dnn_batch = 25
num_batches = n_synth // dnn_batch

#opt_t = keras.optimizers.SGD(learning_rate=inner_lr)
ce_loss = tf.keras.losses.CategoricalCrossentropy()
kl_loss = tf.keras.losses.KLDivergence()

# GAN
batch_size= n_synth  # we'll generate/train on n_synth at once
epochs    = 200
radius = 0.8
y_max = 0.8

opt_g      = keras.optimizers.Adam(5e-4, beta_1=0.5) #1e-4
opt_d      = keras.optimizers.Adam(5e-4, beta_1=0.5) #1e-4
bce        = keras.losses.BinaryCrossentropy()

phys_pen_ratio = 0    # coefficient for physics penalty
gan_ratio = 0.35     # weight for GAN loss
ce_ratio = 0.40
kl_ratio = 1 - phys_pen_ratio - gan_ratio - ce_ratio

print('Phys:',phys_pen_ratio)
print('GAN:', gan_ratio)
print('CE:', ce_ratio)
print('KL:', kl_ratio)

save_dir = "/users/yzhou276/work/dataset_distillation/toptag/distill/"
data_to_save_path = save_dir + f'd_data/{dense_sizes}_dnn_{n_load}_to_{n_synth}.npz'
########## DNN
def build_dense_model(input_dim,
                      dense_sizes,
                      activations,
                      drop_rates,
                      output_dim=2,
                      name='dense_model'):
    x_in = keras.Input(shape=(input_dim,), name=f'{name}_input')
    h = x_in
    w_list = []

    for i, (units, act, dr) in enumerate(zip(dense_sizes, activations, drop_rates)):
        dense = layers.Dense(units, activation=None, name=f'{name}_dense{i}')
        h = dense(h)
        h = layers.Activation(act)(h)
        if dr > 0.:
            h = layers.Dropout(dr)(h)
        w_list.extend(dense.get_weights())   # kernel, bias

    dense_out = layers.Dense(output_dim, activation=None, name=f'{name}_out')
    logits = dense_out(h)
    probs  = layers.Activation('softmax')(logits)
    w_list.extend(dense_out.get_weights())   # add final layer weights

    model = keras.Model(x_in, probs, name=name)
    return model, w_list

@tf.function
def fwd_dense(x, w, dense_sizes, activations, drop_rates, training=True, seed=None):
    h = x
    idx = 0
    for units, act, dr in zip(dense_sizes, activations, drop_rates):
        k, b = w[idx], w[idx+1]; idx += 2
        h = tf.matmul(h, k) + b
        h = tf.keras.activations.get(act)(h)
        if dr > 0. and training:
            h = tf.nn.dropout(h, rate=dr, seed=seed)
    # output layer
    k_out, b_out = w[idx], w[idx+1]
    logits = tf.matmul(h, k_out) + b_out
    return tf.nn.softmax(logits)

@tf.function
def train_dnn_tf(X_tr, Y_tr, X_va, Y_va, w_init, epochs, patience, batch, lr):
    max_epochs= tf.constant(epochs)
    max_wait  = tf.constant(patience)
    n_samples = tf.shape(X_tr)[0]
    steps     = n_samples // batch

    ep        = tf.constant(0)
    wait      = tf.constant(0)
    best_val  = tf.constant(np.inf, dtype=tf.float32)
    w_t       = [tf.convert_to_tensor(w) for w in w_init]
    best_w    = w_t

    best_w    = w_t
    loop_vars = [ep, wait, best_val] + w_t + best_w
    n_w       = len(w_t)
    def cond(ep, wait, best_val, *vars):
        return tf.logical_and(ep < max_epochs,
                              wait < max_wait)

    def body(ep, wait, best_val, *vars):
        w_t    = list(vars[:n_w])
        best_w = list(vars[n_w:])
        ep     = ep + 1

        perm = tf.random.shuffle(tf.range(n_samples))
        X_s  = tf.gather(X_tr, perm)
        Y_s  = tf.gather(Y_tr, perm)

        i = tf.constant(0)
        def inner_cond(i, w_t):
            return i < steps
        def inner_body(i, w_t):
            x_mb = X_s[i*batch:(i+1)*batch]
            y_mb = Y_s[i*batch:(i+1)*batch]
            with tf.GradientTape() as tape:
                tape.watch(w_t)
                yp       = fwd_dense(x_mb, w_t, dense_sizes, activations, drop_rates, training=True)
                loss_mb  = ce_loss(y_mb, yp)
            grads = tape.gradient(loss_mb, w_t)
            w_t   = [wt - lr*g for wt,g in zip(w_t, grads)]
            return i+1, w_t
        _, w_t = tf.while_loop(inner_cond, inner_body,
                               (i, w_t),
                               parallel_iterations=1)
        val_pred = fwd_dense(X_va, w_t, dense_sizes, activations, drop_rates, training=False)
        val_l    = ce_loss(Y_va, val_pred)

        improved = val_l < best_val
        best_val = tf.where(improved, val_l, best_val)
        best_w   = [tf.where(improved, wt, bw)
                    for wt,bw in zip(w_t, best_w)]
        wait     = tf.where(improved, 0, wait + 1)

        return ([ep, wait, best_val] + w_t + best_w)
    final_vars = tf.while_loop(cond, body, loop_vars,
                               parallel_iterations=1)
    best_w_final = final_vars[3 + n_w : 3 + 2*n_w]
    return best_w_final
        

# Train teacher
teacher_model, teacher_w_np = build_dense_model(m*3, dense_sizes, activations, drop_rates, name='teacher')

batch_size_t = 10*dnn_batch

teacher_w = train_dnn_tf(X_tr=X_train_flat, Y_tr=Y_train ,
                         X_va=X_val_flat, Y_va=Y_val,
                         w_init=teacher_w_np,
                         epochs=dnn_epochs, patience=patience_num, batch=batch_size_t,
                         lr=inner_lr)

teacher_train_pred = fwd_dense(X_train_flat, teacher_w, dense_sizes, activations, drop_rates, training=False)
teacher_val_pred = fwd_dense(X_val_flat, teacher_w, dense_sizes, activations, drop_rates, training=False)
teacher_test_pred = fwd_dense(X_test_flat, teacher_w, dense_sizes, activations, drop_rates, training=False)

auc_teacher = roc_auc_score(Y_test[:,1], teacher_test_pred[:,1])
print('Teacher AUC:', auc_teacher)

# Train ref
(X_ref_train, X_ref_val, X_ref_test,
 Y_ref_train, Y_ref_val, Y_ref_test) = data_split(X_train, Y_train, val=int((X_train.shape[0]-n_synth)/2), test=int((X_train.shape[0]-n_synth)/2), shuffle=False)
X_ref_train_flat = X_ref_train.reshape(-1,X_ref_train.shape[1]*X_ref_train.shape[2])
X_ref_val_flat = X_ref_val.reshape(-1,X_ref_val.shape[1]*X_ref_val.shape[2])
X_ref_test_flat = X_ref_test.reshape(-1,X_ref_test.shape[1]*X_ref_test.shape[2])

ref_model, ref_w_np = build_dense_model(m*3, dense_sizes, activations, drop_rates, name='ref')
#ref_w_np = teacher_w_np

ref_w = train_dnn_tf(X_tr=X_ref_train_flat, Y_tr=Y_ref_train ,
                     X_va=X_val_flat, Y_va=Y_val,
                     w_init=ref_w_np,
                     epochs=dnn_epochs, patience=patience_num, batch=dnn_batch,
                     lr=inner_lr)

ref_test_pred = fwd_dense(X_test_flat, ref_w, dense_sizes, activations, drop_rates, training=False)
auc_ref = roc_auc_score(Y_test[:,1], ref_test_pred[:,1])
print('ref AUC:', auc_ref)

# Set up student
#student_model, student_w_np = build_dense_model(m*3, dense_sizes, activations, drop_rates, name='student') # build a student dnn
#student_w_np = teacher_w_np

########## GAN
def phys_transform(raw):
    # raw: (batch, m, 3)
    pt_raw, y_raw, phi_raw = tf.unstack(raw, axis=-1)
    pt  = tf.nn.softplus(pt_raw)           # pT > 0
    '''
    y   = y_max * tf.tanh(y_raw)           # |y| ≤ y_max
    phi = tf.math.sqrt(radius**2 - y**2) * tf.tanh(phi_raw)         # φ ∈ (−π, π)
    '''
    y   = y_max * tf.nn.softsign(y_raw)           # |y| ≤ y_max
    phi = tf.math.sqrt(radius**2 - y**2) * tf.nn.softsign(phi_raw)         # φ ∈ (−π, π)
    return tf.stack([pt, y, phi], axis=-1)

class Generator(keras.Model):
    def __init__(self, n_synth, m, latent_dim):
        super().__init__()
        self.n_synth, self.m, self.latent_dim = n_synth, m, latent_dim
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.d3 = layers.Dense(n_synth * m * 3, activation=None)
        # fixed alternating labels
        half = n_synth // 2
        labels = np.vstack([
            np.tile([0,1], (half,1)),       # quark
            np.tile([1,0], (n_synth-half,1))# gluon
        ]).astype('float32')
        self.labels_const = tf.convert_to_tensor(labels)  # (n_synth,2)

    def call(self, z):
        # z: (batch, latent_dim)
        batch = tf.shape(z)[0]
        x = self.d1(z)
        x = self.d2(x)
        x = self.d3(x)                                   # (batch, n_synth*m*3)
        jets3d = tf.reshape(x, (-1, self.n_synth, self.m, 3))
        jets3d = phys_transform(jets3d)                  # (batch, n_synth, m,3)

        jets_flat = tf.reshape(jets3d, (-1, self.m * 3)) # (batch*n_synth, m*3)

        # repeat & flatten labels
        labels = tf.repeat(self.labels_const[None,...],
                           repeats=batch, axis=0)      # (batch, n_synth,2)
        labels_flat = tf.reshape(labels, (-1, 2))        # (batch*n_synth,2)

        return jets3d, jets_flat, labels_flat


class Discriminator(keras.Model):
    def __init__(self, m):
        super().__init__()
        self.d1  = layers.Dense(128, activation='relu')
        self.d2  = layers.Dense(128, activation='relu')
        self.out = layers.Dense(1,   activation='sigmoid')

    def call(self, inputs):
        # inputs is a list/tuple: [jets_flat, labels_flat]
        jets_flat, labels_flat = inputs
        x = tf.concat([jets_flat, labels_flat], axis=1)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)


gen  = Generator(n_synth, m, latent_dim)
disc = Discriminator(m)


disc.compile(
    optimizer=opt_d,
    loss='binary_crossentropy'
)


########## Train
@tf.function
def train_step(noise, student_w_np):
    """
    One generator update:
      - noise: (1, latent_dim)
    Returns:
      loss_total, loss_gan, phys_penalty  (scalar tensors)
    """
    with tf.GradientTape() as tape:
        # generate
        jets3d, jets_flat, labels_flat = gen(noise, training=True)

        # GAN loss (average D score per noise)
        d_perjet = disc([jets_flat, labels_flat], training=False)  # (1*n_synth,1)
        d_perjet = tf.reshape(d_perjet, (-1, n_synth, 1))         # (1,n_synth,1)
        d_avg    = tf.reduce_mean(d_perjet, axis=1)               # (1,1)
        loss_gan = bce(tf.ones_like(d_avg), d_avg)                # want D_avg → 1

        # DNN train
        student_w = train_dnn_tf(X_tr=jets_flat, Y_tr=labels_flat,
                                 X_va=X_val_flat, Y_va=Y_val,
                                 w_init=student_w_np,
                                 epochs=dnn_epochs, patience=patience_num, batch=dnn_batch,
                                 lr=inner_lr)
        student_test_pred = fwd_dense(X_test_flat, student_w, dense_sizes, activations, drop_rates, training=False)
        loss_ce = ce_loss(Y_test, student_test_pred)
        loss_kl = kl_loss(teacher_test_pred, student_test_pred)

        
        # physics penalty on y, phi edges (|y|>0.9*y_max, |phi|>0.9*pi)
        y_vals   = jets3d[..., 1]
        phi_vals = jets3d[..., 2]

        radius_vals = tf.sqrt(y_vals**2 + phi_vals**2)
        threshold = 1 * radius
        pen_rad   = tf.square(tf.maximum(radius_vals - threshold, 0.))
        phys_penalty = tf.reduce_mean(pen_rad)
        '''
        pen_y    = tf.square(tf.maximum(tf.abs(y_vals) - 0.9*y_max, 0.))
        pen_phi  = tf.square(tf.maximum(tf.abs(phi_vals) - 0.9*np.pi, 0.))
        phys_penalty = alpha * (tf.reduce_mean(pen_y) + tf.reduce_mean(pen_phi))
        '''

        # total generator loss
        loss_total = gan_ratio*loss_gan + ce_ratio*loss_ce + kl_ratio*loss_kl + phys_pen_ratio*phys_penalty

    grads = tape.gradient(loss_total, gen.trainable_variables)
    opt_g.apply_gradients(zip(grads, gen.trainable_variables))

    return loss_total, loss_gan, phys_penalty, loss_ce, loss_kl, student_test_pred, jets3d, jets_flat, labels_flat

best_auc = 0
#noise = tf.zeros((1, latent_dim))
#noise = tf.constant(0.01, shape=(1, latent_dim))
noise = tf.random.normal((1, latent_dim))
for epoch in range(1, epochs+1):
    # (a) sample real batch for D
    idx      = np.random.choice(n_load, n_synth, replace=False)
    Xr_flat  = X_true_flat[idx]     # (n_synth, m*3)
    Yr_flat  = Y_true[idx]          # (n_synth, 2)

    # (b) generate fake batch once for D
    #noise = tf.random.normal((1, latent_dim))
    _, Xf_flat_t, Yf_flat_t = gen(noise, training=False)
    Xf_flat  = Xf_flat_t.numpy()
    Yf_flat  = Yf_flat_t.numpy()

    # (c) update discriminator
    d_loss_r = disc.train_on_batch([Xr_flat, Yr_flat], np.ones((n_synth,1)))
    d_loss_f = disc.train_on_batch([Xf_flat, Yf_flat], np.zeros((n_synth,1)))
    d_loss   = 0.5 * (d_loss_r + d_loss_f)

    # (d) update generator with custom train_step
    student_model, student_w_np = build_dense_model(m*3, dense_sizes, activations, drop_rates, name='student') # build a student dnn
    loss_tot_epoch, loss_gan_epoch, phys_pen_epoch, loss_ce_epoch, loss_kl_epoch, student_test_pred_epoch, jets_epoch, jets_flat_epoch ,labels_epoch = train_step(noise, student_w_np)

    epoch_auc = roc_auc_score(Y_test[:,1], student_test_pred_epoch[:,1])

    print('Epoch', epoch)
    print('Tot loss:', loss_tot_epoch)
    print('GAN loss:', loss_gan_epoch)
    print('CE loss:', loss_ce_epoch)
    print('KL loss:', loss_kl_epoch)
    print('Physics penalty:', phys_pen_epoch)
    print('Student AUC:', epoch_auc)

    
    if epoch_auc > best_auc:
        best_auc = epoch_auc
        jets_epoch = jets_epoch.numpy().squeeze(0)
        labels_epoch = labels_epoch.numpy()
        print('Saving Datasets')
        np.savez_compressed(data_to_save_path, X=jets_epoch, Y=labels_epoch)

        best_epoch = epoch
    '''
    epoch_away = epoch - best_epoch
    if epoch_away == patience_num:
        print('Early Stopping')
        break
    '''


########## Get Datasets
'''
noise     = tf.random.normal((1, latent_dim))
X_synth3d, _, Y_synth_flat = gen(noise, training=False)
X_synth   = X_synth3d.numpy().squeeze(0)  # (n_synth, m, 3)
Y_synth   = Y_synth_flat.numpy()          # (n_synth, 2)
print("Final synthetic jets:", X_synth.shape, "labels:", Y_synth.shape)
np.savez_compressed(data_to_save_path, X=X_synth, Y=Y_synth)
'''

data = np.load(data_to_save_path)
X_3d_synth = data["X"]          # (N, m, 3)
Y_synth    = data["Y"] 
X_flat_synth = X_3d_synth.reshape(-1,X_3d_synth.shape[1]*X_3d_synth.shape[2])
print("flat:", X_flat_synth.shape,  "3-d:", X_3d_synth.shape,  "labels:", Y_synth.shape)
print(X_flat_synth)
print(X_3d_synth)
print(Y_synth)

########## Test on DNN
'''
# Use previous student weights
test_w = train_dnn_tf(X_tr=X_flat_synth, Y_tr=Y_synth ,
                      X_va=X_val_flat, Y_va=Y_val,
                      w_init=student_w_np,
                      epochs=dnn_epochs, patience=patience_num, batch=dnn_batch,
                      lr=inner_lr)
'''
test_model, test_w_np = build_dense_model(m*3, dense_sizes, activations, drop_rates, name='test')
#test_w_np = teacher_w_np

test_w = train_dnn_tf(X_tr=X_flat_synth, Y_tr=Y_synth ,
                      X_va=X_val_flat, Y_va=Y_val,
                      w_init=test_w_np,
                      epochs=dnn_epochs, patience=patience_num, batch=dnn_batch,
                      lr=inner_lr)

test_test_pred = fwd_dense(X_test_flat, test_w, dense_sizes, activations, drop_rates, training=False)
auc_test = roc_auc_score(Y_test[:,1], test_test_pred[:,1])
print('Test AUC:', auc_test)
