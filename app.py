# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv('/Users/siddantprabhudessai/Desktop/spotify/spotify dataset.csv')

# Data Pre-processing
df.dropna(inplace=True)  # Remove missing values
df.drop_duplicates(inplace=True)  # Remove duplicates

# Select relevant features
features = df[['danceability', 'energy', 'valence', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']]

# Data Analysis and Visualizations
# Visualize Distributions
sns.histplot(df['danceability'], kde=True)
plt.title('Danceability Distribution')
plt.show()

sns.boxplot(x=df['danceability'])
plt.title('Danceability Boxplot')
plt.show()

# Correlation Matrix
# Select numeric columns only for correlation matrix
numeric_cols = features.columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Clustering
# Standardize Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# PCA for Dimensionality Reduction (Optional)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

df['Cluster'] = clusters

# Visualize Clusters
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('Clusters of Songs')
plt.show()

# Model Building
# Split Data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['Cluster'], test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Recommendations System
# Predict Cluster for New Data
new_song_data = pd.DataFrame({
    'danceability': [0.5],  # Example value
    'energy': [0.6],        # Example value
    'valence': [0.7],       # Example value
    'tempo': [120],         # Example value
    'loudness': [-5.0],     # Example value
    'speechiness': [0.05],  # Example value
    'acousticness': [0.1],  # Example value
    'instrumentalness': [0.0],  # Example value
    'liveness': [0.15]      # Example value
})
new_song_features = scaler.transform(new_song_data)
predicted_cluster = model.predict(new_song_features)

# Recommend Songs
recommended_songs = df[df['Cluster'] == predicted_cluster[0]]
print("Recommended Songs:\n", recommended_songs[['track_name', 'track_artist', 'playlist_name']])
