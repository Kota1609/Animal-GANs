import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, ks_2samp
import matplotlib.pyplot as plt
from cuml.manifold import TSNE

# File paths
real_data_file = '/home/kota/chat-with-website/RAG/AnimalGAN-main/Data/Data_training.tsv'
generated_data_file = '/home/kota/chat-with-website/RAG/AnimalGAN-main/results/generated_samples.tsv/generated_data_1000.tsv'
output_tsne_file = '/home/kota/chat-with-website/RAG/AnimalGAN-main/results/tsne-gan-3.png'
output_pca_file = '/home/kota/chat-with-website/RAG/AnimalGAN-main/results/pca-gan-3.png'

# Load datasets
real_data = pd.read_csv(real_data_file, sep='\t')
generated_data = pd.read_csv(generated_data_file, sep='\t')

# Ensure both datasets have matching numeric columns
common_columns = set(real_data.columns) & set(generated_data.columns)
real_data = real_data[list(common_columns)].dropna().reset_index(drop=True)
generated_data = generated_data[list(common_columns)].dropna().reset_index(drop=True)

# Select numeric columns
numeric_columns = real_data.select_dtypes(include=[np.number]).columns.intersection(generated_data.columns)

# Statistical tests: Mean, variance, and distribution overlap
stats_summary = []

for col in numeric_columns:
    real_col = real_data[col]
    generated_col = generated_data[col]
    
    # Compute statistics
    mean_real = real_col.mean()
    mean_generated = generated_col.mean()
    var_real = real_col.var()
    var_generated = generated_col.var()
    
    # Perform t-test and KS test
    t_stat, t_p_value = ttest_ind(real_col, generated_col, equal_var=False)
    ks_stat, ks_p_value = ks_2samp(real_col, generated_col)
    
    stats_summary.append({
        'Feature': col,
        'Mean_Real': mean_real,
        'Mean_Generated': mean_generated,
        'Variance_Real': var_real,
        'Variance_Generated': var_generated,
        'T-Test_PValue': t_p_value,
        'KS-Test_PValue': ks_p_value
    })

stats_df = pd.DataFrame(stats_summary)
stats_output_file = '/home/kota/chat-with-website/RAG/AnimalGAN-main/results/statistical_tests_summary.csv'
stats_df.to_csv(stats_output_file, index=False)
print(f"Statistical test summary saved to {stats_output_file}")

# Combine datasets for PCA and t-SNE
real_data['Source'] = 'Real'
generated_data['Source'] = 'Generated'
data_combined = pd.concat([real_data, generated_data])

# Normalize features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_combined[numeric_columns])

# Prepare summary stats string
summary_stats_text = '\n'.join([
    f"Mean Diff: {abs(stats_df['Mean_Real'].mean() - stats_df['Mean_Generated'].mean()):.2f}",
    f"Var Diff: {abs(stats_df['Variance_Real'].mean() - stats_df['Variance_Generated'].mean()):.2f}",
    f"T-Test Avg P-Value: {stats_df['T-Test_PValue'].mean():.4f}",
    f"KS-Test Avg P-Value: {stats_df['KS-Test_PValue'].mean():.4f}"
])

# Perform PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(normalized_data)
pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
pca_df['Source'] = data_combined['Source'].values

# Save PCA plot with stats
plt.figure(figsize=(10, 7))
for source, group in pca_df.groupby('Source'):
    plt.scatter(group['PCA1'], group['PCA2'], label=source, alpha=0.6)
plt.title('PCA Visualization of Real and Generated Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.text(-0.4, 0.4, summary_stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)
plt.grid()
plt.savefig(output_pca_file)
plt.show()
print(f"PCA visualization saved to {output_pca_file}")

# Perform t-SNE
print("Performing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=True)
tsne_results = tsne.fit_transform(normalized_data)

# Add t-SNE results to combined DataFrame
data_combined['TSNE-1'] = tsne_results[:, 0]
data_combined['TSNE-2'] = tsne_results[:, 1]

# Save t-SNE plot with stats
plt.figure(figsize=(10, 7))
for source, group in data_combined.groupby('Source'):
    plt.scatter(group['TSNE-1'], group['TSNE-2'], label=source, alpha=0.6)
plt.title('t-SNE Visualization of Real and Generated Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.text(0.02, 0.95, summary_stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)
plt.grid()
plt.savefig(output_tsne_file)
plt.show()
print(f"t-SNE visualization saved to {output_tsne_file}")
