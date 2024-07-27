from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
import cProfile
import pstats
from pstats import SortKey
from tqdm import tqdm


def load_phrases_from_json(filepath: str) -> list:
    """ load JSON file and create a list of phrases """
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def load_model(sentence_transformer_model: str) -> SentenceTransformer:
    """ Load the sentence transformer model (recommended model: 'paraphrase-MiniLM-L6-v2') """
    model = SentenceTransformer(sentence_transformer_model)
    return model


def embed_phrases(model: SentenceTransformer, phrases: list) -> list:
    """Generate embeddings for a list of phrases."""
    embeddings = model.encode(phrases)
    return embeddings


def cluster_embeddings(embeddings: list, num_clusters: int, rand_state: int) -> list:
    """Cluster embeddings using KMeans."""
    clustering_model = KMeans(n_clusters=num_clusters, random_state=rand_state)
    clustering_model.fit(embeddings)
    return clustering_model.labels_


def group_phrases_by_cluster(phrases: list, cluster_assignments: list, num_clusters: int) -> list:
    """Group phrases by their cluster assignments."""
    clustered_phrases = [[] for _ in range(num_clusters)]
    for phrase, cluster_id in tqdm(zip(phrases, cluster_assignments), desc='group clusters'):
        clustered_phrases[cluster_id].append(phrase)
    return clustered_phrases


def print_clusters(clustered_phrases: list):
    """Print the clustered phrases."""
    for i, cluster in enumerate(clustered_phrases):
        print(f"Cluster {i + 1}:")
        for phrase in cluster:
            print(f"  {phrase}")
        print()


def main(filepath: str, num_clusters: int = 3, rand_state: int = 42):
    phrases = load_phrases_from_json(filepath)   # approximately 13k phrases
    model = load_model('paraphrase-MiniLM-L6-v2')
    embeddings = embed_phrases(model, phrases)
    cluster_assignments = cluster_embeddings(embeddings, num_clusters, rand_state)
    clustered_phrases = group_phrases_by_cluster(phrases, cluster_assignments, num_clusters)
    print_clusters(clustered_phrases)

    print(f"Embeddings created: {len(embeddings)} vectors of size {len(embeddings[0])}")


if __name__ == '__main__':
    with cProfile.Profile() as profile:
        file_path = "../data/accounts_list.json"
        main(file_path, num_clusters=1000, rand_state=42)

    p = pstats.Stats(profile)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats('clustering.py', 6)


