from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed, Memory
import matplotlib.pyplot as plt
from grader import score
import os

# Initialize joblib cache
memory = Memory(location='cache', verbose=0)

def load_and_preprocess_data(file_path, n_features):
    df = pd.read_csv(file_path)
    features = df.iloc[:, 1:n_features+1].values
    # Handle missing values
    if np.any(np.isnan(features)):
        imputer = SimpleImputer(strategy='mean')
        features = imputer.fit_transform(features)
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"Data points: {len(df)}")
    return df['id'].values, features_scaled, df

def visualize_dimensions(features, labels, dim1, dim2, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, dim1-1], features[:, dim2-1], c=labels, cmap='viridis', s=10)
    plt.xlabel(f'Dimension {dim1}')
    plt.ylabel(f'Dimension {dim2}')
    plt.title(f'Cluster Visualization: Dim {dim1} vs Dim {dim2}')
    plt.colorbar(label='Cluster Label')
    plt.savefig(filename)
    plt.close()

def handle_clustering(func):
    """Decorator: Handle clustering errors and K-Means fallback"""
    def wrapper(features, n_clusters, seed, *args, **kwargs):
        np.random.seed(seed)
        try:
            labels = func(features, n_clusters, seed, *args, **kwargs)
            return labels, True
        except Exception as e:
            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=seed
            )
            labels = kmeans.fit_predict(features)
            return labels, True
    return wrapper

@memory.cache
@handle_clustering
def run_kmeans(features, n_clusters, seed):
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=100,
        max_iter=500,
        tol=1e-5,
        random_state=seed
    )
    labels = kmeans.fit_predict(features)
    return labels

@handle_clustering
def run_gmm(features, n_clusters, seed):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=seed)
    kmeans.fit_predict(features)
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='tied',
        max_iter=300,
        random_state=seed,
        init_params='kmeans',
        means_init=kmeans.cluster_centers_
    )
    labels = gmm.fit_predict(features)
    return labels

@handle_clustering
def run_hierarchical_clustering(features, n_clusters, seed):
    max_samples = 10000
    n_samples = len(features)
    if n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        features_sample = features[indices]
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        labels_sample = hierarchical.fit_predict(features_sample)
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(features_sample)
        distances, indices_nn = nn.kneighbors(features)
        labels = labels_sample[indices_nn.flatten()]
    else:
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        labels = hierarchical.fit_predict(features)
    return labels

@memory.cache
@handle_clustering
def run_hybrid_clustering(features, n_clusters, seed):
    # Initial clustering on dimensions 2 and 3
    dim_2_3 = features[:, [1, 2]]
    kmeans_2d = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=seed)
    initial_labels = kmeans_2d.fit_predict(dim_2_3)
    # Perform hierarchical clustering on all dimensions
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward',
        metric='euclidean'
    )
    labels = hierarchical.fit_predict(features)
    return labels

def evaluate_fmi(ids, labels):
    submission = pd.DataFrame({'id': ids, 'label': labels}).sort_values("id").reset_index(drop=True)
    labels_pred = submission["label"].tolist()
    print(f"Evaluating FMI: {len(labels_pred)} labels submitted")
    result = score(labels_pred)
    return result

def save_results(ids, labels, output_file):
    result = pd.DataFrame({'id': ids, 'label': labels})
    result = result.sort_values('id').reset_index(drop=True)
    result.to_csv(output_file, index=False)

def process_seed(seed, public_features, n_clusters, methods, name_len, public_ids):
    np.random.seed(seed)
    results = []
    for method_name, method_func in methods.items():
        labels, success = method_func(public_features, n_clusters=n_clusters, seed=seed)
        if not success:
            print(f"Skipping {method_name} due to failure (seed: {seed}).")
            continue
        score = evaluate_fmi(public_ids, labels)
        if score > 0.9:
            print(f"{method_name} FMI: {' ' * (name_len - len(method_name))}{score:.4f} (seed: {seed})")
        results.append((method_name, labels, score))
    return results

def main():
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Load and preprocess public dataset
    public_ids, public_features, public_df = load_and_preprocess_data('public_data.csv', n_features=4)
    
    # Visualize dimension pairs (2 vs 3)
    raw_features = public_df.iloc[:, 1:5].values
    visualize_dimensions(raw_features, np.zeros(len(raw_features)), 2, 3, 'plots/dim_2_vs_3_initial.png')
    
    # Clustering methods
    methods = {
        'K-Means': run_kmeans,
        'GMM': run_gmm,
        'Hierarchical Clustering': run_hierarchical_clustering,
        'Hybrid Clustering': run_hybrid_clustering
    }
    print("Evaluating clustering methods on public dataset...")
    name_len = max([len(k) for k in methods.keys()])
    
    seeds = [21, 3347, 7499, 8161, 9551, 10331, 15263, 17047, 18181, 18287, 18481]
    
    all_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_seed)(seed, public_features, 15, methods, name_len, public_ids)
        for seed in tqdm(seeds, desc="Processing seeds")
    )
    
    # Find best method and seed
    best_method = None
    best_labels = None
    best_score = -1
    best_seed = -1
    for seed, results in zip(seeds, all_results):
        for method_name, labels, score in results:
            if score > best_score:
                best_score = score
                best_method = method_name
                best_labels = labels
                best_seed = seed
    
    # Save and visualize best results
    sid = "r13922044"
    print(f"Best method for public dataset: {best_method} with FMI Score: {best_score:.4f}, seed: {best_seed}")
    save_results(public_ids, best_labels, f'{sid}_public.csv')
    save_results(public_ids, best_labels, 'public_submission.csv')
    visualize_dimensions(raw_features, best_labels, 2, 3, f'plots/dim_2_vs_3_{best_method.lower().replace(" ", "_")}_best.png')
    
    # Process private dataset
    private_ids, private_features, private_df = load_and_preprocess_data('private_data.csv', n_features=6)
    np.random.seed(best_seed)
    private_labels, success = methods[best_method](private_features, n_clusters=23, seed=best_seed)
    if not success:
        print(f"{best_method} failed for private dataset. Falling back to K-Means.")
        private_labels, success = run_kmeans(private_features, n_clusters=23, seed=best_seed)
    save_results(private_ids, private_labels, f'{sid}_private.csv')
    save_results(private_ids, private_labels, 'private_submission.csv')

if __name__ == '__main__':
    main()