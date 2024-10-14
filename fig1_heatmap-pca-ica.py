#%%
# ----------------------------------------------------------------------------
import os
import scipy
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns; sns.set(style="white")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.transforms import Bbox
from PIL import Image
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
norm1_ica = ica / np.linalg.norm(ica, axis=1).reshape(-1, 1)
norm1_pca = pca / np.linalg.norm(pca, axis=1).reshape(-1, 1)

dict_norm1 = {
    "pca" : norm1_pca,
    "ica" : norm1_ica
}

def topwords_with_thresh(vectype, axis, thresh=1, top_n=10):
    global dict_norm1, count_p, wordlist_org
    vecs = dict_norm1[vectype]
    ax = abs(axis)
    args_ax = np.argsort(vecs[:, ax])[::-1]
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
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.5))
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 7.5))

vmin100 = -0.01
vmax100 = 1
vmin5 = -0.01
vmax5 = 1
cmap_name = "magma_r"

axes = [i for i in range(100)]
words = []
for i in axes:
    words += topwords_with_thresh("pca", i, thresh=100, top_n=4)
ax1_wordids = [get_id[w] for w in words]
vecs_pca = norm1_pca[ax1_wordids]
vecs_pca = vecs_pca[:, axes]
g = sns.heatmap(vecs_pca, ax=ax1, cmap=cmap_name, vmin=vmin100, vmax=vmax100, cbar=False)
ax1.set_title("PCA", fontsize=32, weight='bold', color='red')
ax1.set_yticks([])
ax1.set_xticks([0, 20, 40, 60, 80])
ax1.set_xticklabels([0, 20, 40, 60, 80], fontsize=25, rotation=0)
rect1 = Rectangle((0, 1), 0.05, -0.05, fill=False, color='red', linewidth=3, transform=ax1.transAxes)
ax1.add_patch(rect1)


axes = [i for i in range(100)]
words = []
for i in axes:
    words += topwords_with_thresh("ica", i, thresh=100, top_n=4)
ax2_wordids = [get_id[w] for w in words]
vecs_ica = norm1_ica[ax2_wordids]
vecs_ica = vecs_ica[:, axes]
g = sns.heatmap(vecs_ica, ax=ax2, cmap=cmap_name, vmin=vmin100, vmax=vmax100, cbar=False)
ax2.set_title("ICA", fontsize=32, weight='bold', color='blue')
ax2.set_yticks([])
ax2.set_xticks([0, 20, 40, 60, 80])
ax2.set_xticklabels([0, 20, 40, 60, 80], fontsize=25, rotation=0)
rect2 = Rectangle((0, 1), 0.05, -0.05, fill=False, color='blue', linewidth=3, transform=ax2.transAxes)
ax2.add_patch(rect2)


axes = [i for i in range(5)]
words = []
for i in axes:
    words += topwords_with_thresh("pca", i, thresh=100, top_n=4)
ax3_wordids = [get_id[w] for w in words]
vecs_pca = norm1_pca[ax3_wordids]
vecs_pca = vecs_pca[:, axes]
g = sns.heatmap(vecs_pca, ax=ax3, cmap=cmap_name, vmin=vmin5, vmax=vmax5, cbar=False)
ax3.tick_params(axis='x', labelsize=25)
ax3.set_yticks([])
for i, word in enumerate(words):
    ax3.annotate(word, xy=(0, i+0.5), xytext=(-5, 0), textcoords='offset points', ha='right', va='center', fontsize=27)
rect3 = Rectangle((0, 0), 1, 1, fill=False, color='red', linewidth=6, transform=ax3.transAxes)
ax3.add_patch(rect3)


axes = [i for i in range(5)]
words = []
for i in axes:
    words += topwords_with_thresh("ica", i, thresh=100, top_n=4)
ax4_wordids = [get_id[w] for w in words]
vecs = norm1_ica[ax4_wordids]
vecs = vecs[:, axes]
g = sns.heatmap(vecs, ax=ax4, cmap=cmap_name, vmin=vmin5, vmax=vmax5, cbar=False)
ax4.tick_params(axis='x', labelsize=25)
ax4.set_yticks([])

colors = {
    0 : "#FBB4AE",
    1 : "#B3CDE3",
    2 : "#CCEBC5",
    3 : "#DECBE4",
    4 : "#FED9A6"
}
for i, word in enumerate(words):
    color_idx = i // 4
    ax4.annotate(text=" "*19, xy=(0, i+0.5), xytext=(-5, 0), textcoords='offset points', ha='right', va='center', fontsize=27, bbox=dict(boxstyle="square,pad=0.01", fc=colors[color_idx], ec="none"))

for i, word in enumerate(words):
    ax4.annotate(word, xy=(0, i+0.5), xytext=(-5, 0), textcoords='offset points', ha='right', va='center', fontsize=27)
rect4 = Rectangle((0, 0), 1, 1, fill=False, color='blue', linewidth=6, transform=ax4.transAxes)
ax4.add_patch(rect4)

fig1.tight_layout(pad=0.5)
fig2.tight_layout(pad=0.5)

canvas1 = FigureCanvas(fig1)
canvas2 = FigureCanvas(fig2)
canvas1.draw()
canvas2.draw()

bbox1 = Bbox.from_bounds(0, 0, fig1.get_figwidth(), fig1.get_figheight())
bbox2 = Bbox.from_bounds(0, 0, fig2.get_figwidth(), fig2.get_figheight())
image1 = np.asarray(canvas1.buffer_rgba())
image2 = np.asarray(canvas2.buffer_rgba())

combined_image = Image.new("RGBA", (image1.shape[1], image1.shape[0] + image2.shape[0]))
combined_image.paste(Image.fromarray(image1), (0, 0))
combined_image.paste(Image.fromarray(image2), (0, image1.shape[0]))

fig3, ax = plt.subplots(figsize=(15, 15))
ax.imshow(combined_image)
ax.axis('off')

norm = plt.Normalize(vmin=vmin100, vmax=vmax100)
sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
sm.set_array([])
cbar = fig3.colorbar(sm, ax=ax, fraction=0.0297, pad=0.01, aspect=32)
cbar.ax.tick_params(labelsize=25)

fig3.tight_layout(pad=0.1)

os.makedirs("./figures", exist_ok=True)
fig3.savefig("./figures/fig1_heatmap-pca-ica.png", dpi=300)
# ----------------------------------------------------------------------------