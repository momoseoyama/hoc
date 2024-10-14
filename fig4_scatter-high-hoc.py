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
norm1_ica = ica / np.linalg.norm(ica, axis=1).reshape(-1, 1)

def hoc(mat_X):
    mat_Y = mat_X ** 2
    return np.dot(mat_Y.T, mat_Y) / mat_Y.shape[0]

ec = hoc(ica)
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
def topwords_with_thresh(axis, thresh=100, top_n=10):
    global norm1_ica, count_p, wordlist_org
    ax = abs(axis)
    args_ax = np.argsort(norm1_ica[:, ax])[::-1]
    if axis == 197:
        args_ax = args_ax[::-1]
    words = []
    ii = 0
    while len(words) < top_n:
        if count_p[args_ax[ii]] >= thresh:
            words.append(wordlist_org[args_ax[ii]])
        ii += 1
    return words

def topword_x2y2(x, y, n=10):
    global ica, wordlist_org
    ax_x = ica[:, x]
    ax_y = ica[:, y]
    powp = (ax_x ** 2) * (ax_y ** 2)
    args_powp = np.argsort(powp)[::-1]
    words = [(wordlist_org[j], powp[j]) for j in args_powp]
    res = words[:n]
    return res
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
def draw_ax(fig, gs, position, dict_param):
    global norm1_ica

    ft_title = dict_param["fontsize"]["title"]
    ft_xylabel = dict_param["fontsize"]["xylabel"]
    ft_tick = dict_param["fontsize"]["tick"]
    ft_word = dict_param["fontsize"]["word"]

    ax = fig.add_subplot(gs[position[0], position[1]])

    axis1 = dict_param['axis_pair'][0]
    axis2 = dict_param['axis_pair'][1]
    ax.scatter(norm1_ica[:, axis1], norm1_ica[:, axis2], s=1, c='darkblue')
    ax.set_title(f"$\\mathbf{{ICA}}$  $\\mathrm{{E}}(S_{{{axis1}}}^2 S_{{{axis2}}}^2) = {ec[axis1, axis2]:.3f}$", fontsize=ft_title, y=1.02)
    ax.set_xlabel(f"axis {axis1}", fontsize=ft_xylabel)
    ax.set_ylabel(f"axis {axis2}", fontsize=ft_xylabel)
    ax.tick_params(labelsize=ft_tick)
    ax.set_xlim(-0.4, 0.85)
    ax.set_ylim(-0.4, 0.85)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)

    words_ica1 = topwords_with_thresh(axis1, thresh=100, top_n=4)
    words_ica2 = topwords_with_thresh(axis2, thresh=100, top_n=4)
    words_ica = words_ica1 + words_ica2

    texts_ica = []
    for word in words_ica:
        x, y = norm1_ica[get_id[word], axis1], norm1_ica[get_id[word], axis2]
        text = ax.text(x, y, word, fontsize=ft_word, color="white", bbox=dict(boxstyle="square,pad=0.01", fc="dodgerblue", ec="none", alpha=0.9))
        ax.plot([x, x], [y, y], 'o-', markersize=8, linewidth=2, color='blue')
        texts_ica.append(text)

    words_xy = topword_x2y2(axis1, axis2, n=6)
    texts_xy = []
    for word, val in words_xy:
        x, y = norm1_ica[get_id[word], axis1], norm1_ica[get_id[word], axis2]
        text = ax.text(x, y, word, fontsize=ft_word, color="white", bbox=dict(boxstyle="square,pad=0.01", fc="red", ec="none", alpha=0.7))
        ax.plot([x, x], [y, y], 'o-', markersize=8, linewidth=2, color='red')
        texts_xy.append(text)

    texts = texts_ica + texts_xy
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color='black', lw=2))
    return ax

def create_multi_subplot_figure(n_rows, n_cols, params):
    fig = plt.figure(figsize=(10*n_cols, 10*n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols)
    axes = []
    for i, param in enumerate(params):
        row = i // n_cols
        col = i % n_cols
        ax = draw_ax(fig, gs, (row, col), param)
        axes.append(ax)
    fig.tight_layout()
    return fig, axes
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
n_rows, n_cols = 1, 2

general_param = {
    "fontsize": {
        "title": 48,
        "xylabel": 35,
        "tick": 30,
        "word": 30
        },
    "figsize": (10, 10),
     }

koko_params = [
    {"axis_pair": (10, 2)},
    {"axis_pair": (27, 64)},
]

params = [dict(general_param, **param) for param in koko_params]
fig, axes = create_multi_subplot_figure(n_rows, n_cols, params)
os.makedirs("./figures", exist_ok=True)
fig.savefig(f"./figures/fig4_scatter-high-hoc2.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
n_rows, n_cols = 2, 3

general_param = {
    "fontsize": {
        "title": 48,
        "xylabel": 35,
        "tick": 30,
        "word": 30
        },
    "figsize": (10, 10),
     }

koko_params = [
    {"axis_pair": (10, 2)},
    {"axis_pair": (10, 16)},
    {"axis_pair": (10, 160)},
    {"axis_pair": (27, 11)},
    {"axis_pair": (27, 64)},
    {"axis_pair": (27, 104)},
]

params = [dict(general_param, **param) for param in koko_params]
fig, axes = create_multi_subplot_figure(n_rows, n_cols, params)
os.makedirs("./figures", exist_ok=True)
fig.savefig(f"./figures/fig10_high-hoc6.png", bbox_inches='tight', pad_inches=0.1, dpi=100)
# ----------------------------------------------------------------------------