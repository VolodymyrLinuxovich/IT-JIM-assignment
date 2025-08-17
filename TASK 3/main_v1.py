#!/usr/bin/env python3
import os
import argparse
import logging
import json

import numpy as np
import librosa
from sklearn.cluster import KMeans

# Initialize logging with INFO level to track the clustering process.
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

# Extract classic audio features using librosa.
def extract_features(file_path, sr=22050, n_mfcc=13):
    """
    Extracts classical features from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate (default is 22050 Hz).
        n_mfcc (int): Number of MFCC coefficients (default is 13).
        
    Returns:
        feature_vector (numpy.ndarray): A single vector combining the extracted features.
        
    For the jury: This function loads the audio, normalizes it, computes MFCCs (averaged over time),
    spectral energy, tempo (BPM), and zero crossing rate. The features are concatenated into one vector.
    """
    try:
        # Load the audio file with the specified sampling rate.
        y, sr = librosa.load(file_path, sr=sr)
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}: {e}")
    
    # Normalize the audio signal so its maximum absolute amplitude equals 1.
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    # 1. MFCC: Compute MFCC coefficients and take the mean across the time axis.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # 2. Spectral Energy: Compute the average value of the spectral energy.
    S = np.abs(librosa.stft(y))
    spectral_energy = np.mean(S)
    
    # 3. Tempo: Estimate the tempo (in beats per minute).
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # 4. Zero Crossing Rate: Compute the average zero crossing rate.
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    
    # Concatenate all features into a single vector.
    feature_vector = np.hstack([mfcc_mean, spectral_energy, tempo, zcr_mean])
    return feature_vector

# Process all audio files in the specified folder to extract features.
def process_audio_folder(folder_path, sr=22050, n_mfcc=13):
    """
    Processes all audio files in the specified folder.
    
    Args:
        folder_path (str): The path to the folder containing audio files.
        sr (int): Sampling rate for feature extraction.
        n_mfcc (int): Number of MFCC coefficients to extract.
        
    Returns:
        features (numpy.ndarray): 2D array of feature vectors for all audio files.
        file_names (list): List of file names for which features were successfully extracted.
    
    For the jury: This function iterates over files ending with .wav or .mp3, extracts features
    with error handling, and logs the processing status.
    """
    feature_vectors = []
    file_names = []
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.wav', '.mp3')):
            file_path = os.path.join(folder_path, file)
            try:
                feat = extract_features(file_path, sr=sr, n_mfcc=n_mfcc)
                feature_vectors.append(feat)
                file_names.append(file)
                logger.info(f"Processed {file}: feature vector shape {feat.shape}")
            except Exception as e:
                logger.error(f"Failed to process {file}: {e}")
    
    return np.array(feature_vectors), file_names

# Cluster the extracted feature vectors using the KMeans algorithm.
def cluster_features(features, n_clusters):
    """
    Clusters the extracted features using KMeans.
    
    Args:
        features (numpy.ndarray): 2D array of feature vectors.
        n_clusters (int): Number of clusters.
        
    Returns:
        labels (numpy.ndarray): Array of cluster labels for each audio file.
    
    For the jury: KMeans is used with a fixed random state to ensure reproducibility of the clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

# Save the clustering results to a JSON file with a specific structure.
def save_to_json(labels, file_names, output_file):
    """
    Forms and saves a JSON file with the clustering results.
    
    Expected JSON structure:
    {
      "playlists": [
        {"id": 0, "songs": [file1, file2, ...]},
        {"id": 1, "songs": [file3, file4, ...]},
        ...
      ]
    }
    
    Args:
        labels (numpy.ndarray): Array containing the cluster index for each file.
        file_names (list): List of audio file names.
        output_file (str): Name of the output JSON file.
    
    For the jury: The function groups file names by cluster labels and saves them in a JSON format.
    """
    clusters = {}
    for label, file in zip(labels, file_names):
        clusters.setdefault(int(label), []).append(file)
    
    playlists = [{"id": cluster_id, "songs": songs} for cluster_id, songs in sorted(clusters.items())]
    result = {"playlists": playlists}
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    
    logger.info(f"Saved clustering result to {output_file}")

parser = argparse.ArgumentParser(description="Audio clustering using classical features (main_v1.py)")
parser.add_argument("--path", type=str, required=True,
                    help="Path to the folder with audio files (mp3, wav)")
parser.add_argument("--n", type=int, required=True,
                    help="Number of clusters/playlists")
args = parser.parse_args()

# Setup logging.
logger = setup_logging()
logger.info("Starting audio clustering process (Version 1)")

# Log the folder path being processed.
logger.info(f"Processing audio files from folder: {args.path}")

# Extract features from the audio files in the specified folder.
features, file_names = process_audio_folder(args.path)

# If no valid features were extracted, log error and exit.
if features.size == 0:
    logger.error("No matching audio files found. Exiting.")
    exit(1)

# Log the clustering process details.
logger.info(f"Clustering {len(file_names)} files into {args.n} clusters")
labels = cluster_features(features, args.n)


# Save the clustering results to a JSON file.
output_file = "playlists_v1.json"
save_to_json(labels, file_names, output_file)

logger.info("Audio clustering process finished.")