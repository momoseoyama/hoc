import os
import json
import time
import scipy
import pickle
import random
import openai
import argparse
import numpy as np
from tqdm import tqdm

def load_data():
    get_id = pickle.load(open('./data/text8_sgns/text8_word-to-id.pkl', 'rb'))
    get_word = pickle.load(open('./data/text8_sgns/text8_id-to-word.pkl', 'rb'))
    count_p = np.fromfile('./data/text8_sgns/text8_wordcount', dtype=np.int32)
    pca2_vecs = pickle.load(open(f"./data/ica_data/pca2_20240103_203056.pkl", 'rb'))
    R_ica = pickle.load(open(f"./data/ica_data/R_ica_20240103_203056.pkl", 'rb'))
    ica2_vecs = np.dot(pca2_vecs, R_ica)
    wids = pickle.load(open(f"./data/ica_data/wids_20240103_203056.pkl", 'rb'))
    wordlist_org = np.array([get_word[wid] for wid in wids])
    return get_word, count_p, pca2_vecs, ica2_vecs, wordlist_org

def process_ica(vecs):
    vecs = vecs[:, np.flip(np.argsort(np.abs(scipy.stats.skew(vecs, axis=0))))]
    vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
    return vecs

def normalize_vectors(ica, pca):
    norm1_ica = ica / np.linalg.norm(ica, axis=1).reshape(-1, 1)
    norm1_pca = pca / np.linalg.norm(pca, axis=1).reshape(-1, 1)
    return {"ica": norm1_ica, "pca": norm1_pca}

def topwords_with_thresh(dict_norm1, count_p, wordlist_org, vectype, axis, thresh=100, top_n=10):
    norm1 = dict_norm1[vectype]
    ax = abs(axis)
    args_ax = np.argsort(norm1[:, ax])[::-1]
    words = []
    ii = 0
    while len(words) < top_n:
        if count_p[args_ax[ii]] >= thresh:
            words.append(wordlist_org[args_ax[ii]])
        ii += 1
    return words

def get_top_words(dict_norm1, count_p, wordlist_org, ica, pca):
    top100words_ica = {}
    top100words_pca = {}
    for i in range(ica.shape[1]):
        top100words_ica[i] = topwords_with_thresh(dict_norm1, count_p, wordlist_org, "ica", i, thresh=100, top_n=100)
    for i in range(pca.shape[1]):
        top100words_pca[i] = topwords_with_thresh(dict_norm1, count_p, wordlist_org, "pca", i, thresh=100, top_n=100)
    return top100words_ica, top100words_pca

def relevance_scoring_using_GPTs(list_a, list_b, list_c, list_d, model):
    list_ab = [list_a, list_b]
    list_cd = [list_c, list_d]
    random.shuffle(list_ab)
    random.shuffle(list_cd)
    list_a, list_b = list_ab
    list_c, list_d = list_cd

    ab = f"List pair (A, B): ([{', '.join(list_a)}], [{', '.join(list_b)}])"
    cd = f"List pair (C, D): ([{', '.join(list_c)}], [{', '.join(list_d)}])"

    list_abcd = [ab, cd]
    random.shuffle(list_abcd)
    ab, cd = list_abcd

    prompt = f"""
    Question: 
        You are given 2 list pairs (A, B), (C, D). 
        If one pair is more semantically relevant than the other, answer the pair. 
        If you cannot determine, answer 'XX'.

    {ab}
    {cd}
    
    Output:
        "AB" if (A, B) is more semantically related
        "CD" if (C, D) is more semantically related
        "XX" if equally related, or you can't decide
    Respond with only AB, CD, or XX.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You're a word list relatedness annotator. Compare given two pairs:"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2,
            n=1,
            stop=None,
            temperature=0.0,
        )
        return str(response.choices[0].message['content'])
    except Exception as e:
        print(f"Error: {e}")
        return None

def compute_relevances(axis_pair_1, axis_pair_2, model, filename, embedding, top100words, save_dir, n_words_in_list=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_file_path = os.path.join(save_dir, f"{embedding}_{filename}_{model}_{n_words_in_list}.json")
    if not os.path.exists(json_file_path):
        with open(json_file_path, "w") as json_file:
            json.dump({}, json_file)
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)
    
    for (i, j), (k, l) in tqdm(zip(axis_pair_1, axis_pair_2)):
        if str(((i, j), (k, l))) in json_data:
            continue

        top_words_i = top100words[i][:n_words_in_list]
        top_words_j = top100words[j][:n_words_in_list]
        top_words_k = top100words[k][:n_words_in_list]
        top_words_l = top100words[l][:n_words_in_list]

        score = relevance_scoring_using_GPTs(
            list_a=top_words_i,
            list_b=top_words_j,
            list_c=top_words_k,
            list_d=top_words_l,
            model=model
        )

        if model == "gpt-4":
            time.sleep(2)

        json_data[str(((i, j), (k, l)))] = score
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file)

    return json_data

def hoc(mat_X):
    mat_Y = mat_X ** 2
    return np.dot(mat_Y.T, mat_Y) / mat_Y.shape[0]

def get_top_another(mat_ec, axis, topk=1):
    res = np.argsort(mat_ec[:, axis])[::-1][1:][topk-1]
    return res

def get_bottom_30per(mat_ec, axis):
    kouho = np.argsort(mat_ec[:, axis])[::-1][int(len(mat_ec) * 0.7):]
    res = random.choice(kouho)
    return res

def run_compare_task(embedding, topk, dims, list_gpts, top100words, ec_mat, save_dir):
    pair_top = [(i, get_top_another(ec_mat, i, topk=topk)) for i in range(dims)]
    pair_bottom = [(i, get_bottom_30per(ec_mat, i)) for i in range(dims)]
    for model in list_gpts:
        compute_relevances(pair_top, pair_bottom, model, f"compare_top{topk}_{dims}", embedding, top100words, save_dir, n_words_in_list=5)
        compute_relevances(pair_bottom, pair_top, model, f"rev_compare_top{topk}_{dims}", embedding, top100words, save_dir, n_words_in_list=5)

def get_json_results(model, filename, embedding, save_dir):
    json_file_path = os.path.join(save_dir, f"{embedding}_{filename}_{model}_5.json")
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)
    return json_data

def get_result_summary(json_data):
    count_dict = {
        "AB": 0,
        "CD": 0,
        "XX": 0,
        "None": 0
    }
    for key, value in json_data.items():
        if value is None:
            count_dict["None"] += 1
        else:
            count_dict[value] += 1
    return count_dict

def show_result_summary_average(model, filename, embedding, save_dir):
    json_data = get_json_results(model, filename, embedding, save_dir)
    json_data_rev = get_json_results(model, f"rev_{filename}", embedding, save_dir)
    count_dict = get_result_summary(json_data)
    count_dict_rev = get_result_summary(json_data_rev)
    print(f"Model: {model}, Filename: {filename}, Embedding: {embedding}")
    print(f"Total: {len(json_data)}")
    print(f"TOP: {(count_dict['AB'] + count_dict_rev['CD']) / 2}")
    print(f"BOTTOM: {(count_dict['CD'] + count_dict_rev['AB']) / 2}")
    print(f"Can't Decide: {(count_dict['XX'] + count_dict_rev['XX']) / 2}")
    print()

def run_experiment(args, top100words, ec_mat):
    print(f"Running experiment with {args.embedding} embedding, top-{args.topk}, {args.dims} dimensions")
    print(f"Results will be saved in: {args.save_dir}")
    run_compare_task(args.embedding, args.topk, args.dims, [args.model], 
                     top100words, ec_mat, args.save_dir)

def display_results(args):
    print(f"Displaying results for {args.embedding} embedding, {args.dims} dimensions")
    print(f"Reading results from: {args.save_dir}")
    if args.topk:
        show_result_summary_average(args.model, f"compare_top{args.topk}_{args.dims}", args.embedding, args.save_dir)
    else:
        for topk in range(1, 6):
            show_result_summary_average(args.model, f"compare_top{topk}_{args.dims}", args.embedding, args.save_dir)

def main():
    parser = argparse.ArgumentParser(description="Run experiments or display results")
    parser.add_argument("mode", choices=["run", "display"], help="Mode to run: 'run' for experiments, 'display' for results")
    parser.add_argument("--embedding", choices=["ICA", "PCA"], help="Embedding type")
    parser.add_argument("--topk", type=int, help="Top-k value for comparison")
    parser.add_argument("--dims", type=int, default=100, help="Number of dimensions to use")
    parser.add_argument("--model", default="gpt-4o-mini", help="GPT model to use")
    parser.add_argument("--save_dir", default="./results", help="Directory to save/load results")
    args = parser.parse_args()

    if args.mode == "run":
        if not all([args.embedding, args.topk]):
            parser.error("The 'run' mode requires --embedding and --topk arguments")

        get_word, count_p, pca2_vecs, ica2_vecs, wordlist_org = load_data()
        ica = process_ica(vecs=ica2_vecs)
        pca = pca2_vecs
        dict_norm1 = normalize_vectors(ica, pca)
        top100words_ica, top100words_pca = get_top_words(dict_norm1, count_p, wordlist_org, ica, pca)
        ec_ica = hoc(ica)
        ec_pca = hoc(pca)
        dict_ec = {"ICA": ec_ica, "PCA": ec_pca}
        
        run_experiment(args, 
                       top100words_ica if args.embedding == "ICA" else top100words_pca, 
                       dict_ec[args.embedding])
    elif args.mode == "display":
        display_results(args)

if __name__ == "__main__":
    main()