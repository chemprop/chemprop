import torch
from tqdm import trange
from sklearn.cluster import MiniBatchKMeans
from .data import MoleculeDataset

def get_cluster_labels(encodings, n_clusters=10000, seed=0, logger=None):
    n_clusters = int(min(n_clusters, len(encodings)/10)) # so we don't crash if we only picked a small number of encodings
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = kmeans.fit_predict(encodings)
    return cluster_labels

def generate_unsupervised_cluster_labels(model, data, args, logger=None):
    encodings = []
    for i in trange(0, len(data), args.batch_size):
        batch = MoleculeDataset(data[i:i + args.batch_size])
        with torch.no_grad():
            encodings.extend([enc for enc in model.encoder(batch.smiles()).cpu().numpy()])
    cluster_labels = get_cluster_labels(encodings, n_clusters=args.unsupervised_n_clusters, logger=logger)
    cluster_labels = cluster_labels.reshape(-1, 1).astype(int).tolist()
    data.set_targets(cluster_labels)

    