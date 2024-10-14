#%%
# ----------------------------------------------------------------------------
import os
import scipy
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set(style="white")
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
pca = pca2_vecs

def hoc(mat_X):
    mat_Y = mat_X ** 2
    return np.dot(mat_Y.T, mat_Y) / mat_Y.shape[0]

ec = hoc(ica)

norm1_ica = ica / np.linalg.norm(ica, axis=1).reshape(-1, 1)
norm1_pca = pca / np.linalg.norm(pca, axis=1).reshape(-1, 1)

dict_norm1 = {
    "pca" : norm1_pca,
    "ica" : norm1_ica
}

def topwords_with_thresh(vectype, axis, thresh=100, top_n=10):
    global dict_norm1, count_p, wordlist_org
    ax = abs(axis)
    norm1 = dict_norm1[vectype]
    args_ax = np.argsort(norm1[:, ax])[::-1]
    words = []
    ii = 0
    while len(words) < top_n:
        if count_p[args_ax[ii]] >= thresh:
            words.append(wordlist_org[args_ax[ii]])
        ii += 1
    return words
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

axis1 = 10
axis2 = 20

ft_title = 48
ft_xylabel = 36
ft_tick = 30
ft_word = 36

ax1.scatter(norm1_pca[:, axis1], norm1_pca[:, axis2], s=1, c='darkred')
ax1.set_title('PCA', fontsize=ft_title, fontweight='bold', y=1.02)
ax1.set_xlabel(f'axis {axis1}', fontsize=ft_xylabel)
ax1.set_ylabel(f'axis {axis2}', fontsize=ft_xylabel)
ax1.set_xlim(-0.4, 0.85)
ax1.set_ylim(-0.4, 0.85)
ax1.tick_params(labelsize=ft_tick)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)

words_pca1 = topwords_with_thresh("pca", axis1, thresh=100, top_n=3)
words_pca2 = topwords_with_thresh("pca", axis2, thresh=100, top_n=3)
words_pca = words_pca1 + words_pca2

ax2.scatter(norm1_ica[:, axis1], norm1_ica[:, axis2], s=1, c='darkblue')
ax2.set_title(f"$\\mathbf{{ICA}}$  $\\mathrm{{E}}(S_{{{axis1}}}^2 S_{{{axis2}}}^2) = {ec[axis1, axis2]:.3f}$", fontsize=ft_title, fontweight='bold', y=1.02)
ax2.set_xlabel(f'axis {axis1}', fontsize=ft_xylabel)
ax2.set_ylabel(f'axis {axis2}', fontsize=ft_xylabel)
ax2.set_xlim(-0.4, 0.85)
ax2.set_ylim(-0.4, 0.85)
ax2.tick_params(labelsize=ft_tick)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)

words_ica1 = topwords_with_thresh("ica", axis1, thresh=100, top_n=5)
words_ica2 = topwords_with_thresh("ica", axis2, thresh=100, top_n=5)
words_ica = words_ica1 + words_ica2

texts_ica = []
for word in words_ica:
    x, y = norm1_ica[get_id[word], axis1], norm1_ica[get_id[word], axis2]
    text = ax2.text(x, y, word, fontsize=ft_word, color="white", bbox=dict(boxstyle="square,pad=0.01", fc="dodgerblue", ec="none", alpha=0.9))
    ax2.plot([x, x], [y, y], 'o-', markersize=8, linewidth=2, color='blue')
    texts_ica.append(text)

adjust_text(texts_ica, ax=ax2, arrowprops=dict(arrowstyle="-", color='dodgerblue', lw=2))
fig.tight_layout(pad=0.5)
os.makedirs("./figures", exist_ok=True)
fig.savefig(f"./figures/fig2_scatter-pca-ica.png", dpi=300, bbox_inches='tight')
# ----------------------------------------------------------------------------