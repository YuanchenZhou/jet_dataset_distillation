import numpy as np
import matplotlib.pyplot as plt
import energyflow as ef

dense_sizes = (100,)*2
n = 2500

n_load = 50000
n_synth = 5000

#data_to_save_path = f'/users/yzhou276/work/dataset_distillation/qgtag/distill/d_data/{dense_sizes}dnn_{n}_ttag.npz'
#data_to_save_path = f'/users/yzhou276/work/dataset_distillation/qgtag/distill/d_data/test_gan.npz' 
data_to_save_path = f'/users/yzhou276/work/dataset_distillation/qgtag/distill/d_data/{dense_sizes}_dnn_{n_load}_to_{n_synth}.npz'


data = np.load(data_to_save_path)
X_3d_synth = data["X"]          # (N, m, 3)
Y_synth    = data["Y"]

print("3-d:", X_3d_synth.shape,  "labels:", Y_synth.shape)

#Copied from top tagging scripts
X_synth_qcd = [] # [1,0] being gluon here
X_synth_top = [] # [0,1] being quark here

for x in range(Y_synth.shape[0]):
    if Y_synth[x][1] == 0:
        X_synth_qcd.append(X_3d_synth[x])
    else:
        X_synth_top.append(X_3d_synth[x])

print(len(X_synth_qcd))
print(len(X_synth_top))

X_synth_qcd = np.array(X_synth_qcd)
X_synth_top = np.array(X_synth_top)

plt.figure(figsize=(8.4,7.2))
plt.scatter(X_synth_qcd[0,:,1],X_synth_qcd[0,:,2],c=X_synth_qcd[0,:,0],cmap = 'plasma', label='Synth Gluon Jet Particles')
plt.ylabel('Phi', fontsize=18)
plt.xlabel('y', fontsize=18)
cbar = plt.colorbar(label='p_t')
cbar.set_label('p_t', fontsize=18)
cbar.ax.tick_params(labelsize=12)
plt.title('Synth Gluon Jet', fontsize=20)
plt.legend()
plt.savefig('/users/yzhou276/work/dataset_distillation/qgtag/plots/jet_plots/synth_gluon.jpg')

plt.figure(figsize=(8.4,7.2))
plt.scatter(X_synth_top[4,:,1],X_synth_top[4,:,2],c=X_synth_top[0,:,0],cmap = 'plasma', label='Synth Quark Jet Particles')
plt.ylabel('Phi', fontsize=18)
plt.xlabel('y', fontsize=18)
cbar = plt.colorbar(label='p_t')
cbar.set_label('p_t', fontsize=18)
cbar.ax.tick_params(labelsize=12)
plt.title('Synth Quark Jet', fontsize=20)
plt.legend()
plt.savefig('/users/yzhou276/work/dataset_distillation/qgtag/plots/jet_plots/synth_quark.jpg')


synth_qcd_jet_mass = [ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X_synth_qcd]
synth_top_jet_mass = [ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X_synth_top]

plt.figure(figsize=(8, 6))
plt.hist(synth_qcd_jet_mass, bins=np.arange(0, 180, 2), label='Synth Gluon', alpha=0.5)
plt.hist(synth_top_jet_mass, bins=np.arange(0, 180, 2), label='Synth Quark', alpha=0.5)
plt.ylabel('Number of jets', fontsize=18)
plt.xlabel('Jet Mass (GeV)', fontsize=18)
plt.title('Synth Dataset Jet Mass', fontsize=20)
plt.legend()
plt.show()
plt.savefig('/users/yzhou276/work/dataset_distillation/qgtag/plots/jet_property/synth_jet_mass.jpg')

# Jet width plot
def calc_jet_width(X): # X is an array of jets with each jet's elements being its particles, and each particle will have [p_t, y, phi]
  jet_width = []
  for x in X:
    mask = x[:,0] > 0
    y_jet = np.sum(x[mask,1]*x[mask,0])/np.sum(x[mask,0])
    phi_jet = np.sum(x[mask,2]*x[mask,0])/np.sum(x[mask,0])
    delta_y = x[mask,1] - y_jet
    delta_phi = x[mask,2] - phi_jet
    delta_R = np.sqrt(delta_y**2 + delta_phi**2)
    W = np.sum(x[mask,0]*delta_R)/np.sum(x[mask,0])
    jet_width.append(W)
  return np.array(jet_width)

quark_jet_width = calc_jet_width(X_synth_top)
gluon_jet_width = calc_jet_width(X_synth_qcd)

plt.figure(figsize=(8, 6))
plt.hist(gluon_jet_width, bins=np.arange(0, 0.01, 0.00005), label='Synth Gluon', alpha=0.5)
plt.hist(quark_jet_width, bins=np.arange(0, 0.01, 0.00005), label='Synth Quark', alpha=0.5)
plt.ylabel('Number of jets', fontsize=18)
plt.xlabel('Jet Width', fontsize=18)
plt.title('Synth Dataset Jet Width', fontsize=20)
plt.legend()
plt.show()
plt.savefig('/users/yzhou276/work/dataset_distillation/qgtag/plots/jet_property/synth_jet_width.jpg')
