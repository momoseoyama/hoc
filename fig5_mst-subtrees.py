#%%
# ----------------------------------------------------------------------------
import os
import scipy
import pickle
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
GET_ID = pickle.load(open('./data/text8_sgns/text8_word-to-id.pkl', 'rb'))
GET_WORD = pickle.load(open('./data/text8_sgns/text8_id-to-word.pkl', 'rb'))
count_p = np.fromfile('./data/text8_sgns/text8_wordcount', dtype=np.int32)
pca2_vecs = pickle.load(open(f"./data/ica_data/pca2_20240103_203056.pkl", 'rb'))
R_ica = pickle.load(open(f"./data/ica_data/R_ica_20240103_203056.pkl", 'rb'))
ica2_vecs = np.dot(pca2_vecs, R_ica)
WIDS = pickle.load(open(f"./data/ica_data/wids_20240103_203056.pkl", 'rb'))
WIDS = np.array(WIDS)
mat_X = np.fromfile('./data/text8_sgns/text8_sgns-Win_ep100').reshape(len(count_p), -1)
mat_cX1 = mat_X - np.mean(mat_X, axis=0)

def process_skew(vecs):
    """
    1. axis is sorted in descending order by the abs(skewness)
    2. skewness <- abs(skewness)
    """
    vecs = vecs[:, np.flip(np.argsort(np.abs(scipy.stats.skew(vecs, axis=0))))]
    vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
    return vecs

ica = process_skew(vecs=ica2_vecs)
pca = pca2_vecs

def hoc(mat_X):
    mat_Y = mat_X ** 2
    return np.dot(mat_Y.T, mat_Y) / mat_Y.shape[0]

ec_ica = hoc(ica)
ec_pca = hoc(pca)

dict_ec = {
    "pca": ec_pca,
    "ica": ec_ica
}

norm1_ica = ica / np.linalg.norm(ica, axis=1).reshape(-1, 1)
norm1_pca = pca / np.linalg.norm(pca, axis=1).reshape(-1, 1)

dict_norm1 = {
    "pca": norm1_pca,
    "ica": norm1_ica
}

def topwords_with_thresh(vectype, axis, count_thresh=100, top_n=10):
    global dict_norm1, count_p, GET_WORD
    normalized_vec = dict_norm1[vectype]
    ax = abs(axis)
    args_ax = np.argsort(normalized_vec[:, ax])[::-1]
    words = []
    ii = 0
    while len(words) < top_n:
        if count_p[args_ax[ii]] >= count_thresh:
            words.append(GET_WORD[args_ax[ii]])
        ii += 1
    return words
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
# node selection
# ----------------------------------------------------------------------------
def l2dist(vec1, vec2):
    return (np.linalg.norm(vec1 - vec2))

def get_dict_intruder_candidates(vectype):
    """
    output: dict_intruder_candidates[ax] = [intruder candidates of axis=ax]
    """
    global dict_norm1, GET_WORD
    vec_norm1 = dict_norm1[vectype]
    dim = vec_norm1.shape[1]
    word_count = vec_norm1.shape[0]

    dict_lower_half = {}
    dict_top10p = {}
    print("2/3 : lower_half / top10p")
    for i in tqdm(range(dim)):
        sorted_indices = np.argsort(vec_norm1[:, i])[::-1]
        mid_point = word_count // 2
        top_10_percent = word_count // 10
        dict_lower_half[i] = set(GET_WORD[wid] for wid in sorted_indices[mid_point:])
        dict_top10p[i] = set(GET_WORD[wid] for wid in sorted_indices[:top_10_percent])

    dict_intruder_candidates = {}
    print("3/3 : candidates")
    for ax in tqdm(range(dim)):
        lower_half = dict_lower_half[ax]
        candidates = set()
        for i in range(dim):
            candidates.update(lower_half & dict_top10p[i])
        dict_intruder_candidates[ax] = sorted(list(candidates))
    return dict_intruder_candidates

class Intruder:
    def __init__(self, count_thresh, top_n):
        global ica, pca, topwords_with_thresh, get_dict_intruder_candidates
        self.dict_topwords = {
            "ica": {i: topwords_with_thresh("ica", i, count_thresh=count_thresh, top_n=top_n) for i in tqdm(range(ica.shape[1]))},
            "pca": {i: topwords_with_thresh("pca", i, count_thresh=count_thresh, top_n=top_n) for i in tqdm(range(pca.shape[1]))}
        }
        self.dict_intruder_candidates = {
            "ica": get_dict_intruder_candidates("ica"),
            "pca": get_dict_intruder_candidates("pca")
        }
        self.dim = ica.shape[1]
    
    def score(self, vectype, axis):
        return self._inter_dist(vectype, axis) / self._intra_dist(vectype, axis)
    
    def average_scores(self, vectype, n_runs=100):
        scores = []
        for i in tqdm(range(self.dim)):
            average_i = 0
            for _ in range(n_runs):
                average_i += self.score(vectype, i)
            scores.append(average_i / n_runs)
        return scores
    
    def _inter_dist(self, vectype, axis):
        global mat_cX1, GET_ID
        axis_words = self.dict_topwords[vectype][axis]
        intruder_kouho = self.dict_intruder_candidates[vectype][axis]
        intruder = random.choice(intruder_kouho)
        d = 0
        for w1 in axis_words:
            d += l2dist(mat_cX1[GET_ID[w1]], mat_cX1[GET_ID[intruder]])
        d /= len(axis_words)
        return d
    
    def _intra_dist(self, vectype, axis):
        global mat_cX1, GET_ID
        axis_words = self.dict_topwords[vectype][axis]
        d = 0
        for w1 in axis_words:
            for w2 in axis_words:
                if w1 != w2:
                    d += l2dist(mat_cX1[GET_ID[w1]], mat_cX1[GET_ID[w2]])
        d /= (len(axis_words) * (len(axis_words) - 1))
        return d

intruder_5 = Intruder(100, 5)
random.seed(0)
ica_ave = intruder_5.average_scores("ica", n_runs=100)
nodes_to_show = np.argsort(ica_ave)[::-1][:150]
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
# MST
# ----------------------------------------------------------------------------
def transform_weights(G, method='invert', epsilon=1e-6):
    G_transformed = G.copy()
    if method == 'invert':
        for (u, v, d) in G_transformed.edges(data=True):
            G_transformed[u][v]['weight'] = 1 / (d['weight'] + epsilon)
    elif method == 'negative_log':
        for (u, v, d) in G_transformed.edges(data=True):
            G_transformed[u][v]['weight'] = -np.log(d['weight'] + epsilon)
    else:
        raise ValueError("Invalid method. Choose 'invert' or 'negative_log'")
    return G_transformed

def nodes_to_completegraph(nodes, mat_ec):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in G.nodes:
        for j in G.nodes:
            if i < j:
                G.add_edge(i, j, weight=1/(mat_ec[i, j]))
    return G

def get_mst_inv(G):
    mst = nx.minimum_spanning_tree(G)
    mst_inv = transform_weights(mst, method='invert')
    return mst_inv

G = nodes_to_completegraph(nodes_to_show, dict_ec["ica"])
mst_inv = get_mst_inv(G)
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
# visualization
# ----------------------------------------------------------------------------
def node2cluster(node, clusters):
    for cluster in clusters:
        if node in cluster:
            return cluster

def draw_labels(pos, labels, dict_cluster_color, font_size=10):
    for node, label in labels.items():
        x, y = pos[node]
        cluster = node2cluster(node, dict_cluster_color.keys())
        backcolor, mojicolor = dict_cluster_color[cluster]
        plt.text(x, y, label, fontsize=font_size, ha='center', va='center', color=mojicolor,
                    bbox=dict(
                        facecolor=backcolor, edgecolor="none", 
                        boxstyle='round,pad=0.2', alpha=0.7))

def draw_graph(G, dict_cluster_color, dict_name={"vec": "ica", "graph": "mst"}, font_size=12):
    fig = plt.figure(figsize=(16, 24))
    pos = nx.kamada_kawai_layout(G)
    labels = {i: f"{i}: {topwords_with_thresh(vectype=dict_name['vec'], axis=i, count_thresh=100, top_n=1)[0]}" for i in list(G.nodes)}
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='b', alpha=0, label=labels)
    c = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=2, alpha=1, edge_color=c, edge_cmap=plt.cm.Blues)
    draw_labels(pos, labels, dict_cluster_color, font_size=font_size)
    plt.axis('off')
    plt.subplots_adjust(left=-0.06, right=1.05, top=1.08, bottom=-0.08)
    os.makedirs("./figures", exist_ok=True)
    plt.savefig(f"./figures/fig12_mst150.pdf", bbox_inches='tight')

def gen_backcolors(num_colors):
    colors = []
    for i in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors

def gen_mojicolors(backcolors):
    mojicolors = []
    for backcolor in backcolors:
        r, g, b = tuple(int(backcolor.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        if r*0.299 + g*0.587 + b*0.114 > 186:
            mojicolors.append("#000000")
        else:
            mojicolors.append("#ffffff")
    return mojicolors

def get_clusters(G, n_clusters=10):
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', n_init=10)
    labels = sc.fit_predict(nx.to_numpy_array(G))
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(list(G.nodes)[i])
    return clusters

random.seed(1)
np.random.seed(1)
clusters = get_clusters(mst_inv, n_clusters=10)
backcolors = gen_backcolors(len(clusters))
mojicolors = gen_mojicolors(backcolors)
dict_cluster_color = {tuple(cl): (backcolors[i], mojicolors[i]) for i, cl in enumerate(clusters)}
draw_graph(mst_inv, dict_cluster_color, dict_name={"vec": "ica", "graph": "mst_inv_ica"}, font_size=11)
# ----------------------------------------------------------------------------


#%%
# ----------------------------------------------------------------------------
# subgraphs
# ----------------------------------------------------------------------------
def draw_graph_2(G, dict_cluster_color, clusters_to_draw, dict_name, font_size):
    nodes_to_draw = clusters_to_draw[0] + clusters_to_draw[1]
    G_sub = G.subgraph(nodes_to_draw)
    fig = plt.figure(figsize=(16, 4))
    pos = nx.kamada_kawai_layout(G_sub)
    labels = {i: f"{i}: {topwords_with_thresh(vectype=dict_name['vec'], axis=i, count_thresh=100, top_n=1)[0]}" for i in list(G_sub.nodes)}
    nx.draw_networkx_nodes(G_sub, pos, node_size=300, node_color='b', alpha=0, label=labels)
    c = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
    nx.draw_networkx_edges(G_sub, pos, width=2, alpha=1, edge_color=c, edge_cmap=plt.cm.Blues)
    draw_labels(pos, labels, {k: v for k, v in dict_cluster_color.items() if set(k) & set(nodes_to_draw)}, font_size=font_size)
    plt.axis('off')
    plt.subplots_adjust(left=-0.05, right=1.05, top=1.06, bottom=-0.06)
    os.makedirs("./figures", exist_ok=True)
    plt.savefig(f"./figures/fig5_mst-subtree_{dict_name['graph']}.pdf", bbox_inches='tight')

clusters_to_draw = [clusters[1], clusters[9]]
draw_graph_2(mst_inv, dict_cluster_color, clusters_to_draw, dict_name={"vec": "ica", "graph": "greek"}, font_size=13)

clusters_to_draw = [clusters[4], clusters[5]]
draw_graph_2(mst_inv, dict_cluster_color, clusters_to_draw, dict_name={"vec": "ica", "graph": "dna"}, font_size=13)
# ----------------------------------------------------------------------------
# %%
