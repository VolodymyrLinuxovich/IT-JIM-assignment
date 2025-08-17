# Audio Clustering Tool

## Overview

Audio Clustering Tool is a project designed to automatically group music files into playlists based on their audio characteristics. Two approaches are implemented:
- **Classical Features:** This version uses handcrafted features such as MFCCs, spectral energy, tempo, and zero-crossing rate computed via Librosa.
- **Deep Learning Features:** This version employs a pretrained Wav2Vec2 model from Torchaudio to extract deep feature embeddings, which are then clustered.

Both versions are CLI-compatible and output the clustering results in a structured JSON file.

## Features

- **Feature Extraction (Classical):**  
  Uses Librosa to extract:
  - **MFCCs:** Computes 13 MFCC coefficients averaged over time.
  - **Spectral Energy:** Computes the average STFT magnitude.
  - **Tempo:** Estimates beats per minute (BPM).
  - **Zero Crossing Rate:** Calculates the average crossing rate of the signal.
  
- **Feature Extraction (Deep Learning):**  
  Uses Torchaudio's pretrained Wav2Vec2 model to compute embeddings from audio files.
  
- **Clustering:**  
  Utilizes the KMeans algorithm to group audio files into user-specified clusters (playlists).

- **JSON Output:**  
  Saves the clustering results in a JSON file with the following structure:

{
    "playlists": [
        {
            "id": 0,
            "songs": [
                "0fa4cfa4-14d0-4850-88b0-8d21382edadb.mp3",
                "356a199c-513f-4656-a0b0-f12e2610bfee.mp3",
                "53c96f2d-18e4-4226-9efc-a6a92815faac.mp3",
                "6577d5cd-87cc-4cc3-af04-640960c36168.mp3",
                "93e444d8-69d3-47b9-b00b-c83e3f6af448.mp3",
                "a008b71f-8d0b-4843-acfd-1dd8443dcfa8.mp3",
                "a703d5a3-db4d-4891-aee5-41af16a20783.mp3",
                "f2a71b03-8a8a-4824-8b9b-27032b10b15f.mp3"
            ]
        },
        {
            "id": 1,
            "songs": [
                "897cfb82-42ed-4304-be7a-07dd38d2b1ea.mp3",
                "9f9236d2-54dd-4d16-b551-f2a562a3299c.mp3",
                "c4068c84-2903-411b-a50d-c2b2de5d2a49.mp3"
            ]
        },
        {
            "id": 2,
            "songs": [
                "456a0433-9e97-400c-987c-f633a8a8f3ff.mp3",
                "57f508ff-f78a-42cf-b5f4-d2f15bd53b78.mp3",
                "80db79fa-f030-4c7b-9360-3edaf0e6c1bd.mp3",
                "9eef63e1-0b16-4f8f-95d4-6ca93f2db91e.mp3"
            ]
        }
    ]
}

# Project Structure

project_root/
├── data/  
│   └── (Audio files to be processed)
├── main_v1.py
├── main_v2.py
├── playlists_v1.json
├── playlists_v2.json
├── requirements.txt
└── README.md


## Requirements

- **Python Libraries:**
  - **numpy:** For numerical operations.
  - **librosa:** For audio processing and feature extraction.
  - **scikit-learn:** For clustering (KMeans).
  - **torch:** For deep learning computations.
  - **torchaudio:** For loading the pretrained audio model.

See the [requirements.txt] file for the exact list of dependencies.

# Audio Clustering Tool Documentation

# 1. Clone the Repository:
#    ```bash
#    git clone <repository_url>
#    cd project_root
#    ```

# 2. Install Dependencies:
#    Install all required packages using pip:
#    ```bash
#    pip install -r requirements.txt
#    ```

# 3. Directory Setup:
#    Ensure your project directory is organized as described above, with your audio files placed in the appropriate folder.

# ## Process Flow

# ### Input Loading:
# The tool accepts a folder path (via CLI argument --path) containing audio files in .wav or .mp3 formats.

# ### Feature Extraction:
# Based on the version:
# - **Classical Approach:** Uses Librosa to compute MFCCs, spectral energy, tempo, and zero crossing rate.
# - **Deep Learning Approach:** Uses the pretrained Wav2Vec2 model to compute deep embeddings.

# ### Clustering:
# The extracted features (fixed-length vectors) are clustered using the KMeans algorithm to produce the desired number of clusters (specified via --n).

# ### Output Generation:
# The clustering results are saved in a JSON file:
# - For the classical approach: playlists_v1.json
# - For the deep learning approach: playlists_v2.json

# ## Result Verification

# After processing, verify the output by opening the JSON file to see a list of playlists,
# where each playlist contains the filenames of audio files belonging to that cluster.
# This organized output can be used for further analysis or integration into a music library application.

# ## Code Modules

# ### Hand-Crafted Features Module (main_v1.py)
# - **extract_features:**  
#   Loads each audio file, normalizes the signal, computes MFCCs, spectral energy, tempo, and zero crossing rate, 
#   and concatenates these into one feature vector.
# - **process_audio_folder:**  
#   Iterates over all audio files in the specified folder, extracts features with error handling, and logs processing status.
# - **cluster_features:**  
#   Uses KMeans to cluster the audio files based on the extracted features.
# - **save_to_json:**  
#   Saves the clustering results into a JSON file with the required structure.

# ### Deep Learning Features Module (main_v2.py)
# - **extract_dl_features:**  
#   Loads each audio file using Torchaudio, ensures proper sampling rate, converts multi-channel audio to mono,
#   passes it through the Wav2Vec2 model, and averages the resulting embeddings.
# - **process_audio_folder:**  
#   Processes all audio files similarly to the classical approach but using the deep learning feature extractor.
# - **cluster_features:**  
#   Clusters the deep features using KMeans.
# - **save_to_json:**  
#   Outputs the clustering result in a JSON file.

# ### CLI Interface:
# Both versions accept the following arguments:
# - **--path:** Specifies the directory containing audio files.
# - **--n:** Specifies the number of clusters to generate.

# #### Example:
# ```bash
# python main_v1.py --path ./data/ --n 3
# python main_v2.py --path ./data/ --n 3
# ```