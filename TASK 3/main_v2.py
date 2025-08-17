#!/usr/bin/env python3
import os
import logging
import json
import argparse

import torch
import torchaudio
import numpy as np
from sklearn.cluster import KMeans

def setup_logging():
    # Initialize logging (INFO level) to trace the clustering process.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def load_dl_model():
    # Load the pretrained Wav2Vec2 model and its configuration from torchaudio.
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    mdl = bundle.get_model()  # Get the DL model for feature extraction.
    mdl.eval()                # Set the model to evaluation mode.
    sr = bundle.sample_rate   # Retrieve the expected sample rate.
    return mdl, sr

def extract_dl_features(fp, mdl, sr, dev='cpu'):
    """
    Extracts deep learning features from an audio file using Wav2Vec2.
    
    fp  : Path to the audio file.
    mdl : Pretrained model for feature extraction.
    sr  : Expected sample rate for the model.
    dev : Device for computation ('cpu' or 'cuda').

    Returns:
         A fixed-length feature vector obtained by averaging over the time axis.
    """
    # Load the audio file.
    wav, fs = torchaudio.load(fp)
    if fs != sr:
        # Resample if the sample rate does not match.
        wav = torchaudio.functional.resample(wav, fs, sr)
    if wav.shape[0] > 1:
        # Convert multi-channel audio to mono by averaging channels.
        wav = torch.mean(wav, dim=0, keepdim=True)
    wav = wav.to(dev)
    with torch.no_grad():
        out = mdl(wav)  # Process the audio through the model.
        # Some models may return a tuple; extract the first element if so.
        feats = out[0] if isinstance(out, tuple) else out
    # Compute the mean over the time dimension to obtain a fixed-length vector.
    emb = feats.mean(dim=1).squeeze().cpu().numpy()
    return emb

def process_audio_folder(pth, mdl, sr, dev='cpu'):
    """
    Processes all audio files in the given folder to extract their features.
    
    pth   : Directory path containing audio files.
    mdl   : Model for feature extraction.
    sr    : Expected sample rate.
    dev   : Device for computation.
    
    Returns:
            feats   - 2D array of feature vectors.
            fnames  - List of processed file names.
    """
    
    feats = []
    fnames = []
    for f in os.listdir(pth):
        if f.lower().endswith(('.wav', '.mp3')):
            fp = os.path.join(pth, f)
            try:
                feat = extract_dl_features(fp, mdl, sr, dev=dev)
                feats.append(feat)
                fnames.append(f)
                logger.info(f"Processed {f}: feature vector shape {feat.shape}")
            except Exception as e:
                logger.error(f"Failed to process {f}: {e}")
    return np.array(feats), fnames

def cluster_features(feats, n_cls):
    """
    Clusters the extracted feature vectors using the KMeans algorithm.
    
    feats  : 2D array of feature vectors.
    n_cls  : Number of clusters.
    
    Returns:
             An array of cluster labels for each audio file.
    """
    km = KMeans(n_clusters=n_cls, random_state=42)
    labels = km.fit_predict(feats)
    return labels

def save_to_json(labels, fnames, out_file):
    """
    Saves the clustering results in JSON format.

    JSON structure:
        {
          "playlists": [
             {"id": 0, "songs": [file1, file2, ...]},
             {"id": 1, "songs": [file3, file4, ...]},
             ...
          ]
        }
    
    labels   : Array of cluster labels.
    fnames   : List of audio file names.
    out_file : Output JSON file name.
    """
    clust = {}
    for lab, f in zip(labels, fnames):
        clust.setdefault(int(lab), []).append(f)
    
    plist = [{"id": cid, "songs": songs} for cid, songs in sorted(clust.items())]
    res = {"playlists": plist}
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
    logger.info(f"Saved clustering result to {out_file}")

# Process CLI arguments for flexibility.
parser = argparse.ArgumentParser(description="Audio clustering using DL features")
parser.add_argument("--path", type=str, required=True,
                    help="Path to the folder with audio files (mp3, wav)")
parser.add_argument("--n", type=int, required=True,
                    help="Number of clusters/playlists")
args = parser.parse_args()

# Set the folder path and number of clusters based on CLI input.
pth = args.path
n_cls = args.n

logger = setup_logging()
logger.info("Starting audio clustering process with DL features")
logger.info(f"Processing audio files from folder: {pth}")

# Set device for computation (preferably 'cuda' if available).
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {dev}")

# Load the deep learning model and move it to the selected device.
mdl, sr = load_dl_model()
mdl.to(dev)

# Extract features from all audio files in the specified folder.
feats, fnames = process_audio_folder(pth, mdl, sr, dev=dev)
if feats.size == 0:
    logger.error("No audio files found for processing. Exiting.")
    exit(1)

logger.info(f"Clustering {len(fnames)} files into {n_cls} clusters")
labels = cluster_features(feats, n_cls)

# Save the clustering results in JSON format.
out_file = "playlists_v2.json"
save_to_json(labels, fnames, out_file)

logger.info("Audio clustering process finished.")