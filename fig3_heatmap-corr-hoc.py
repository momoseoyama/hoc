#%%
# ----------------------------------------------------------------------------
import os
import scipy
import pickle
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
get_id = pickle.load(open('./data/text8_sgns/text8_word-to-id.pkl', 'rb'))
get_word = pickle.load(open('./data/text8_sgns/text8_id-to-word.pkl', 'rb'))
count_p = np.fromfile('./data/text8_sgns/text8_wordcount', dtype=np.int32)
pca2_vecs = pickle.load(open(f"./data/ica_data/pca2_20240103_203056.pkl", 'rb'))
R_ica = pickle.load(open(f"./data/ica_data/R_ica_20240103_203056.pkl", 'rb'))
ica2_vecs = np.dot(pca2_vecs, R_ica)
wids = pickle.load(open(f"./data/ica_data/wids_20240103_203056.pkl", 'rb'))
wordlist_org = np.array([get_word[wid] for wid in wids])

def process_ica(vecs):
    """
    1. axis is sorted in descending order by the abs(skewness)
    2. skewness <- abs(skewness)
    """
    vecs = vecs[:, np.flip(np.argsort(np.abs(scipy.stats.skew(vecs, axis=0))))]
    vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
    return vecs

ica = process_ica(vecs=ica2_vecs)

def hoc(mat_X):
    mat_Y = mat_X ** 2
    return np.dot(mat_Y.T, mat_Y) / mat_Y.shape[0]

ec = hoc(ica)
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
color = sns.color_palette("Blues", as_cmap=True)

co = np.corrcoef(ica.T)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
fig = plt.figure(figsize=(15, 6.5))

ax1 = fig.add_subplot(gs[0])
g = sns.heatmap(co, cmap=color, vmin=0, vmax=1, ax=ax1, cbar_kws={'pad': 0.01})
ax1.set_title(r"$\mathrm{E}(S_i S_j)$", fontsize=36, y=1.02)
ax1.set_xlabel(r"$i$", fontsize=20)
ax1.set_ylabel(r"$j$", fontsize=20)
ax1.set_xticks([0, 45, 90, 135, 180, 225, 270])
ax1.set_xticklabels([0, 45, 90, 135, 180, 225, 270], fontsize=20, rotation=0)
ax1.set_yticks([0, 45, 90, 135, 180, 225, 270])
ax1.set_yticklabels([0, 45, 90, 135, 180, 225, 270], fontsize=20, rotation=0)
for t in g.collections[0].colorbar.ax.get_yticklabels():
    t.set_fontsize(20)

ax2 = fig.add_subplot(gs[1])
g2 = sns.heatmap(ec, cmap=color, vmin=1, vmax=2.5, ax=ax2, cbar_kws={'pad': 0.01})
ax2.set_title(r"$\mathrm{E}(S_i^2 S_j^2)$", fontsize=36, y=1.02)
ax2.set_xlabel(r"$i$", fontsize=20)
ax2.set_ylabel(r"$j$", fontsize=20)
ax2.set_xticks([0, 45, 90, 135, 180, 225, 270])
ax2.set_xticklabels([0, 45, 90, 135, 180, 225, 270], fontsize=20, rotation=0)
ax2.set_yticks([0, 45, 90, 135, 180, 225, 270])
ax2.set_yticklabels([0, 45, 90, 135, 180, 225, 270], fontsize=20, rotation=0)
for t in g2.collections[0].colorbar.ax.get_yticklabels():
    t.set_fontsize(20)

os.makedirs("./figures", exist_ok=True)
fig.savefig(f"./figures/fig3_heatmap-corr-hoc.png", dpi=300, bbox_inches='tight')
# ----------------------------------------------------------------------------