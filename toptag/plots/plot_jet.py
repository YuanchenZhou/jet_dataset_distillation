import numpy as np
import matplotlib.pyplot as plt
import energyflow as ef

dense_sizes = (100,)*2
n = 2500

n_load = 50000
n_synth = 5000

#data_to_save_path = f'/users/yzhou276/work/dataset_distillation/toptag/distill/d_data/{dense_sizes}dnn_{n}_ttag.npz'
#data_to_save_path = f'/users/yzhou276/work/dataset_distillation/toptag/distill/d_data/test_gan.npz' 
data_to_save_path = f'/users/yzhou276/work/dataset_distillation/toptag/distill/d_data/{dense_sizes}_dnn_{n_load}_to_{n_synth}.npz'


data = np.load(data_to_save_path)
X_3d_synth = data["X"]          # (N, m, 3)
Y_synth    = data["Y"]

print("3-d:", X_3d_synth.shape,  "labels:", Y_synth.shape)

X_synth_qcd = [] # [1,0]
X_synth_top = [] # [0,1]

for x in range(Y_synth.shape[0]):
    if Y_synth[x][1] == 0:
        X_synth_qcd.append(X_3d_synth[x])
    else:
        X_synth_top.append(X_3d_synth[x])

print(len(X_synth_qcd))
print(len(X_synth_top))

X_synth_qcd = np.array(X_synth_qcd)
X_synth_top = np.array(X_synth_top)

plt.figure(figsize=(7,6))
plt.scatter(X_synth_qcd[0,:,1],X_synth_qcd[0,:,2],c=X_synth_qcd[0,:,0],cmap = 'plasma', label='Synth QCD Jet Particles')
plt.ylabel('Phi', fontsize=18)
plt.xlabel('y', fontsize=18)
cbar = plt.colorbar(label='p_t')
cbar.set_label('p_t', fontsize=18)
cbar.ax.tick_params(labelsize=12)
plt.title('Synth QCD Jet', fontsize=20)
plt.legend()
plt.savefig('/users/yzhou276/work/dataset_distillation/toptag/plots/jet_plots/synth_qcd.jpg')

plt.figure(figsize=(7,6))
plt.scatter(X_synth_top[4,:,1],X_synth_top[4,:,2],c=X_synth_top[0,:,0],cmap = 'plasma', label='Synth TOP Jet Particles')
plt.ylabel('Phi', fontsize=18)
plt.xlabel('y', fontsize=18)
cbar = plt.colorbar(label='p_t')
cbar.set_label('p_t', fontsize=18)
cbar.ax.tick_params(labelsize=12)
plt.title('Synth Top Jet', fontsize=20)
plt.legend()
plt.savefig('/users/yzhou276/work/dataset_distillation/toptag/plots/jet_plots/synth_top.jpg')


synth_qcd_jet_mass = [ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X_synth_qcd]
synth_top_jet_mass = [ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X_synth_top]

plt.figure(figsize=(8, 6))
plt.hist(synth_qcd_jet_mass, bins=np.arange(0, 1000, 8), label='Synth QCD', alpha=0.5)
plt.hist(synth_top_jet_mass, bins=np.arange(0, 1000, 8), label='Synth Top', alpha=0.5)
plt.ylabel('Number of jets', fontsize=18)
plt.xlabel('Jet Mass (GeV)', fontsize=18)
plt.title('Synth Dataset Jet Mass', fontsize=20)
plt.legend()
plt.show()
plt.savefig('/users/yzhou276/work/dataset_distillation/toptag/plots/jet_mass/synth_jet_mass.jpg')
