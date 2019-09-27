''' Perform experiments related to clustering hidden states of an 
LSTM language model. '''

import argparse
import json
import pickle
import time
import random
from collections import Counter

import hdbscan
import unidecode
import numpy as np
import dominate.tags as dt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from nltk import word_tokenize
from nltk.tag.stanford import StanfordPOSTagger

from utils import colour_text as ct
from utils.measurements import get_stats, get_stats_minimal, get_whitespace_stats, get_perplexity
from charlstm import *

# Command-line args =============================================
parser = argparse.ArgumentParser(description="Script for finding, saving, and measuring statistics about clusters.")

parser.add_argument("--action", default="none",
    choices=["get_comp_data", "get_clusters","visualize","make_graph","none","snippet_viz","get_word_vecs","interactive","perplexity","make_word_graph"],
    help="Which action to perform.  default: get_comp_data") 
parser.add_argument("--computation-data-file", default="temp/data.pickle",
    help="The pickle file containing data about the network's computation (cell states etc.) (default 'temp/data.pickle')")
parser.add_argument("--sample-text", default="temp/text.txt",
    help="The file containing th text to run the model on.  (default 'temp/text.txt')")
parser.add_argument("--model-dir", default="saved_models/WAP",
    help="The directory of the saved model being investigated.  (default 'saved_models/WAP')")
parser.add_argument("--cluster-data-file", default="temp/clusters.json",
    help="The JSON file containing a record of clusters found by hdbscan")
parser.add_argument("--clusterer-file", default="temp/clusterer.pickle",
    help="The pickle file in which to store the clusterer.")
parser.add_argument("--embeddings-file", default="temp/embeddings.pickle",
    help="The pickle file in which to store word embeddings.")
parser.add_argument("--text-cutoff", default=30000, type=int,
    help="The maximum number of characters to read from the text sample")

parser.add_argument("--minimal", action="store_true",
    help="Whether to compute only the bare minimum amount of data (i.e. only output gate values)")
parser.add_argument("--whitespace-only", action="store_true",
    help="Whether to only compute/use data for spaces and newlines.")
parser.add_argument("--do-POS", action="store_true",
    help="Whether to calculate POS statistics for clusters.")
parser.add_argument("--layer", default=2, type=int,
    help="Which layer to look at (indexed from 0). Default: 2")


parser.add_argument("--min-cluster-size", default=100, type=int,
    help="The minimum cluster size for HDBSCAN. Default: 100")
parser.add_argument("--min-samples", default=10, type=int,
    help="The 'min samples' parameter for HDBSCAN. Default: 10")
parser.add_argument("--cluster-by", default="output_gate",
    help="Name of property to cluster by. Default: output_gate")

args = parser.parse_args()

# Some useful functions =====================================================

def get_comp_data(model, sample_text, verbose=True, minimal=False, whitespace=False,layer=0):
    if minimal:
        dat = get_stats_minimal(model, sample_text, layer=layer, 
            verbose=verbose)
        return {"output_gate":dat}
    elif whitespace:
        dat, indices, hidden, cell = get_whitespace_stats(model, sample_text, 
            layer=layer, verbose=verbose)
        return {"output_gate": dat.numpy(), "indices": indices, 
            "hidden_states": hidden.numpy(), "cell_states": cell.numpy()}
    else:
        result = get_stats(model, sample_text, layer=layer, 
            verbose=verbose, use_loss=False, fast_mode=True)
        property_names = ["cell_states", "hidden_states", "outputs", 
            "saliencies", "output_gate"]
        return {property_names[i]:p.tolist() for i, p in enumerate(result)}

def get_clusters(data, cluster_by, min_cluster_size=100, min_samples=10):
    ''' Cluster data using hdbscan.  Data must have a field with the
    name given by 'cluster_by' '''
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
        min_samples=min_samples,prediction_data=True)
    clusterer.fit(data[cluster_by])

    labels = clusterer.labels_
    num_clusters = int(labels.max() + 1)
    num_unclustered = len([l for l in labels if l == -1])
    return [int(l) for l in labels], num_clusters, num_unclustered, clusterer

def visualize(data, text, clusterer, cluster_by, whitespace_only=False):
    ''' Visualize appearances of each cluster in the sample text.
    'data' should be output like that from get_comp_data'''

    # Get data for text to use for clustering:
    tokens = [unidecode.unidecode(char) for char in text]
    tokens = [ct.tweak_whitespace(w) for w in tokens[0:-1]]
    
    print("Starting labelling")
    labels, _ = hdbscan.approximate_predict(clusterer, data[cluster_by])
    clusters = sorted(list(set(labels)))
    print("Done labelling")

    content = dt.div(dt.h1("Cluster Visualization"))
    for i, c in enumerate(clusters):
        print("DEBUG: - Visualizing cluster %d" % i)
        details = dt.details(dt.summary("Cluster: %d" %c))
        colors = [1.0 if c == labels[j] else 0.0 for j in range(len(tokens))] 
        if whitespace_only:
            for j in range(len(tokens)):
                colors[j] = colors[j] if text[j] in [" ","\n"] else 0.0
        token_data = zip(tokens, colors)
        details.add(ct.colored_text(token_data))
        content.add(details)
    return content

def snippet(text, i, k=21):
    ''' Get a k-character snippet from the text, ideally centred on index i,
    with the character at index i surrounded by square brackets.
    Example: "said the d[o]ctor to th" is a size-21 snippet  '''
    if i - k//2 < 0:
        return text[:i] + "[%s]"%text[i] + text[i+1:k]
    elif i + k//2 > len(text):
        return text[-k:i] + "[%s]"%text[i] + text[i+1:-1]
    else:
        return text[i-k//2:i] + "[%s]"%text[i] + text[i+1:i+k//2]

def snippet_viz(data, text, clusterer, cluster_by, whitespace_only=False):
    ''' Visualize appearances of each cluster in the sample text.
    'data' should be output like that from get_comp_data'''

    # Get data for text to use for clustering:
    tokens = [unidecode.unidecode(char) for char in text]
    tokens = [ct.tweak_whitespace(w) for w in tokens[0:-1]]
    
    print("Starting labelling")
    labels, _ = hdbscan.approximate_predict(clusterer, data[cluster_by])
    clusters = sorted(list(set(labels)))
    print("Done labelling")

    # Get snippets for each cluster.
    snippets = {cluster:[] for cluster in clusters}
    for i in range(len(data[cluster_by])):
        corresponding_index = i 
        if "indices" in data:
            corresponding_index = data["indices"][i]
        if text[corresponding_index] in [" ","\n"] or not whitespace_only:
            # Get a text "snippet"
            snip = unidecode.unidecode(snippet(text, corresponding_index, k=30))
            snippets[labels[i]].append(snip)

    if args.do_POS:
        print("Starting POS statistics")
        tagger = get_tagger()
        tagged_text = tagger.tag(word_tokenize(text))
        POS_stats = {cluster:Counter() for cluster in clusters}
        POS_stats["total"] = Counter()

        for i in range(len(data[cluster_by])):
            corresponding_index = i 
            if "indices" in data:
                corresponding_index = data["indices"][i]
            if text[corresponding_index] in [" ","\n"] or not whitespace_only:
                prev_word = get_prev_tag(tagged_text,text,corresponding_index)
                tag = prev_word[1]
                POS_stats[labels[i]][tag] += 1
                POS_stats["total"][tag] += 1
            if i % 1000 == 0:
                print("Done up to index %d" % corresponding_index)
        print("Done POS Statistics")


    content = dt.div(dt.h1("Cluster Summary"))
    for k, v in snippets.items():
        print("DEBUG: - Visualizing cluster %s" % k)
        details = dt.details(dt.summary("Cluster: %s" % k))
        random.shuffle(v) # Randomize list of snippets
        if len(v) > 300:
            v = v[:300] # Shorten list of snippets to manageable size
        for snip in v:
            details.add(dt.div(snip))
        if args.do_POS:
            # Summarize statistics
            summary = ""
            total_instances = sum(POS_stats[k].values())
            for tag, count in POS_stats[k].items():
                total_tag_count = POS_stats["total"][tag]
                precision = (count*1.0)/total_instances
                recall = (count*1.0)/total_tag_count
                percentage = 1.0*count/total_tag_count
                summary += "%s: %d (Precision: %.4f, Recall: %.4f)\n" % (tag, count, precision, recall)
            details.add(dt.pre(summary))
        content.add(details)
    return content


def get_avg_representations(data, text, rep_key="hidden_states"):
    ''' For each word in the data set, find what the hidden state after
    that word looks like on average. 
   
    data - the computation data (assumed to include the "indices" data)
    text - the corresponding text
    rep_key - the key in the data to use as representations
    '''
    vocab = set([w.lower() for w in word_tokenize(text)])
    emb_size = len(data[rep_key][0])
    raw_embeddings = {word:[] for word in vocab}
    for i in range(len(data[rep_key])):
        corresponding_index = i 
        if "indices" in data:
            corresponding_index = data["indices"][i]
        if text[corresponding_index] in [" ","\n"]:
            prev_word = get_prev_word(text,corresponding_index).lower()
            raw_embeddings[prev_word].append(np.array(data[rep_key][i]))
        if i % 1000 == 0:
            print("Done up to index %d" % corresponding_index)

    # Now we calculate averages.
    # Note: Not all words actually end up with a corresponding representation
    #  since we're only looking at whitespace, so we need to check before
    #  computing averages.
    average_embeddings = {}
    for word in vocab:
        if len(raw_embeddings[word]) > 0:
            mat = np.array(raw_embeddings[word])
            average_embeddings[word] = np.average(mat,axis=0)
    return average_embeddings, raw_embeddings
    

def make_graph(data, clusterer, cluster_by="hidden_states",k=8,do_pos=False,
    get_labels=False,max_nodes=5000):
    ''' data should contain the field cluster_by and a field 'text' '''
    text = data['text']
    n = len(data[cluster_by])
    if n > max_nodes:
        n = max_nodes
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    print("Created Nodes")

    if get_labels:
        print("Getting clusters (this might take a while)")
        labels, _ = hdbscan.approximate_predict(clusterer, data[cluster_by])
        clusters = sorted(list(set(labels)))
        print("Got clusters")

    tagged_text = None
    if do_pos:
        # Get part-of-speech tags to visualize on graph
        tagger = get_tagger()
        print("Getting POS tags:")
        tagged_text = tagger.tag(word_tokenize(text))
        print("Done POS tagging")

    # Add attributes to nodes:
    print("Adding attributes to nodes:")
    for i in range(n):            
        corresponding_index = i 
        if "indices" in data:
            corresponding_index = data["indices"][i]
        if get_labels:
            G.node[i]["cluster"] = int(labels[i])
        G.node[i]["snippet"] = snippet(text, corresponding_index, k=21)
        G.node[i]["word"] = get_prev_word(text,corresponding_index)
        if do_pos:
            G.node[i]["POS"] = get_prev_tag(tagged_text,text,corresponding_index)[1]
        if i % 10000 == 0:
            print("Done first %d nodes" %i)

    matrix=np.array(data[cluster_by][:n])
    similarities = cosine_similarity(matrix)
    print("Calculated cosine similarity")

    # Add nearest neighbours as edges
    edges = []
    for i in range(n):
        simil = similarities[i]
        neighbour_indices = np.argpartition(simil, (-k+1))[-(k+1):-1]
        for j in neighbour_indices:
            edges.append((i,j))
    G.add_edges_from(edges,edge_type="neighbour")
    # Add an edge from each node to the following node (chronologically)
    #G.add_edges_from([(i,i+1) for i in range(n-1)], edge_type="successor")
    return G

def embeddings_knn(word,embeddings,k=10,metric="cosine"):
    words = list(embeddings.keys())
    X = np.array([embeddings[word] for word in words])
    nbrs = NearestNeighbors(n_neighbors=k,metric=metric).fit(X)
    if type(word) == str:
        inp = [embeddings[word]]
    else:
        # Should be the raw vector
        inp = word
    _, indices = nbrs.kneighbors(inp)
    return [words[i] for i in indices[0]]

def get_tagger():
    ''' Set up & return the Stanford Tagger object.'''
    path_to_model = "/home/avery/Applications/stanford-postagger-2018-02-27/models/english-bidirectional-distsim.tagger"
    path_to_jar = "/home/avery/Applications/stanford-postagger-2018-02-27/stanford-postagger.jar" 
    tagger = StanfordPOSTagger(path_to_model, path_to_jar)
    tagger.java_options = "-mx8192m"
    # Use: tagger.tag(word_tokenize(string)) 
    return tagger

def get_prev_tag(tagged_text,text,i):
    ''' Return the token and tag of the word preceding index i
        Note: this is O(n) in terms of the length of the text, 
        which is kinda silly but actually makes more sense than 
        the most obvious "better" method (i.e. chopping off the last sentence
        of so and then taggin it),
        due to how long the Stanford tagger takes on small samples (due to
        the startup time of the JVM, I think). '''
    tokens = word_tokenize(text[:i])
    index = len(tokens) - 1 # Index of preceding word in the tokenized list.
    return tagged_text[index]

def get_prev_word(text, i):
    #  There's probably a less silly way to do this.
    tokens = word_tokenize(text[i-min(i,100):i])
    # Note: If the text begins with whitespace, we just give the first few
    #  characters a period as their "previous word".
    #  It's not totally ideal, but the period appears so many times in the text
    #   (and also has similar semantics to "start of text") that it shouldn't
    #   affect the word embeddings in any noticeable way whatsoever.
    return tokens[-1] if len(tokens) > 0 else "."

def embeddings_to_text_format(embeddings):
    # Convert embeddings to the text format used for wordvectors.org
    vocab = list(embeddings.keys())
    lines = []
    for word in vocab:
        datastring = " ".join([str(x) for x in embeddings[word]])
        line = "%s %s" % (word, datastring)
        lines.append(line)
    return "\n".join(lines)

# Load data etc. ==================================================

def load_model():
    return CharLSTM.load_with_info(args.model_dir) 

def load_sample_text():
    with open(args.sample_text,"r") as f:
        sample_text = f.read()
    if len(sample_text) > args.text_cutoff:
        sample_text = sample_text[:args.text_cutoff]
    return sample_text

def load_computation_data():
    with open(args.computation_data_file,"rb") as f:
        data = pickle.load(f)
    return data

def load_clusterer():
    with open(args.clusterer_file,"rb") as f:
        data = pickle.load(f)
    return data

# Main script ==============================================================
if __name__=="__main__":
    if args.action == "interactive":
        # (meant to be used in interactive mode (python -i)
        # Simply loads the model, sample text, word embeddings, etc. 
        #  for manual experimentation.
        try:
            sample_text = load_sample_text()
            tokens = word_tokenize(sample_text)
            tokens = [t.lower() for t in tokens]
            word_count = Counter()
            for t in tokens:
                word_count[t] += 1
        except:
            print("Specified text file does not exist")

        try:
            model = load_model()
        except:
            print("Specified model does not exist")

        try:
            data = load_computation_data()
        except:
            print("Specified computation data file does not exist")

        try:
            with open(args.embeddings_file,"rb") as f:
                embeddings, raw_embeddings = pickle.load(f)
                vocab = embeddings.keys()
                singletons = [word for word in vocab 
                    if len(raw_embeddings[word]) == 1]
                num_appearances = {word:len(v) 
                    for word, v in raw_embeddings.items()}
                # Ignore situations where two words separated by an em-dash
                #  are incorrectly combined into one word (not that the
                #  hidden state at this point usually reflects what's expected
                #  after the second word, more or less.)
                dash = '—'
                nodash_embeddings = {word: v 
                    for word, v in embeddings.items() if dash not in word}

        except:
            print("Embeddings file does not exist or failed to load.")

        # Example of something to try out in interactive mode:
        #embeddings_knn("prince",embeddings,k=10):

    if args.action == "get_comp_data":
        # Run the model on a sample text and save the computed hidden states,
        #  output gate activations, etc.
        sample_text = load_sample_text()
        model = load_model()
        
        data = get_comp_data(model, sample_text, verbose=True, 
            minimal=args.minimal, whitespace=args.whitespace_only,
            layer=args.layer)
        data["text"] = sample_text

        with open(args.computation_data_file,"wb") as f:
            pickle.dump(data,f)

    if args.action == "get_word_vecs":
        # Create word vectors as described in the paper.
        data = load_computation_data()
        sample_text = data["text"]

        embeddings, raw_embeddings = get_avg_representations(data, sample_text, 
            rep_key="hidden_states")
        with open(args.embeddings_file,"wb") as f:
            # Note: My OS hung for a while writing this
            #  in a case where the final file size ended up being ~= 6 GB
            pickle.dump( (embeddings,raw_embeddings) ,f)

    if args.action == "get_clusters":
        ''' Create the hdbscan clusterer. '''
        data = load_computation_data()
        print("Starting clustering")
        start_time = time.time()
        
        labels, n, uncl, clusterer = get_clusters(data,args.cluster_by,
            min_cluster_size=args.min_cluster_size, 
            min_samples=args.min_samples)
        elapsed = time.time() - start_time
        print("Found %d clusters with %d points unclustered"%(n, uncl))
        print("Total time elapsed (s): %f" % elapsed)
        with open(args.clusterer_file,"wb") as f:
            pickle.dump(clusterer, f)

    if args.action == "visualize":
        ''' Create a visualization for each cluster which colours certain
        characters of a text according to whether they fall into the cluster.

        Note: The resulting file is pretty large and not very illuminating.
        Use snippet_viz instead. '''
        sample_text = load_sample_text()
        model = load_model()
        clusterer = load_clusterer()

        data = get_comp_data(model, sample_text, 
            verbose=True, minimal=args.minimal,layer=args.layer)
        html = visualize(data, sample_text, clusterer, 
            args.cluster_by, whitespace_only=args.whitespace_only) 
        with open("temp/text_viz.html","w") as f:
            f.write(str(html))

    if args.action == "snippet_viz":
        ''' For each cluster, show a list of up to 300 snippets of text
        surrounding characters falling into that cluster.
        
        Can also measure part-of-speech statistics, but this is slow.'''
        sample_text = load_sample_text()
        model = load_model()
        clusterer = load_clusterer()

        # We compute the data again since you might want to visualize
        #  and/or compute statistics using a different chunk of text
        #  than the one you trained the clusters on.
        data = get_comp_data(model, sample_text, verbose=True, 
            minimal=args.minimal,whitespace=args.whitespace_only)
        html = snippet_viz(data, sample_text, clusterer, args.cluster_by,
            whitespace_only=args.whitespace_only) 
        with open("temp/snippet_viz.html","w") as f:
            f.write(str(html))

    if args.action == "make_graph":
        # Again, we compute data rather than using the precomputed data,
        #  in case you want to visualize using a different text.
        sample_text = load_sample_text()
        model = load_model()
        data = get_comp_data(model, sample_text, 
            verbose=False, minimal=args.minimal)
        data['text'] = sample_text

        clusterer = load_clusterer()
        G = make_graph(data, clusterer, cluster_by=args.cluster_by)
        nx.write_graphml(G,"graph.graphml")

    if args.action == "make_word_graph":
        # Again, we compute data rather than using the precomputed data,
        #  in case you want to visualize using a different text.
        data = load_computation_data()
        clusterer = load_clusterer()
        G = make_graph(data, clusterer, cluster_by=args.cluster_by,do_pos=True)
        nx.write_graphml(G,"graph.graphml")

    if args.action == "perplexity":
        # Just measure the performance of the model on a text,
        #  since I neglected to do this when training.
        sample_text = load_sample_text()
        model = load_model()

        tokens = word_tokenize(sample_text)
        num_words = len(word_tokenize(sample_text))
        # Average word length
        avg_word_length = (len(sample_text[:-1])*1.0)/num_words
        # Average not including spaces:
        #  (This would give lower perplexity, but that would be misleading)
        avg2 = np.average(list(map(lambda x: float(len(x)), tokens)))

        cross_entropy, bpc, ppl = get_perplexity(sample_text, model, 
            avg_word_length)
        print("Cross entropy: %.4f" % cross_entropy)
        print("Bits-per-character: %.4f" % bpc)
        print("Equivalent perplexity per word: %.4f" % ppl)

