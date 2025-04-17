"""1. Validate Your Data Pipeline

2. Implement Data Augmentation

3. Build a Baseline Model

4. Fine-Tune the Model

5. Evaluate the Model

6. Save the Model
"""

import os
import pandas as pd
import librosa
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def validate_audio_file(audio_path: str, sample_rate: int) -> bool:
    """
    Validate if an audio file exists and can be loaded.
    
    Args:
        audio_path (str): Path to the audio file
        sample_rate (int): Target sample rate for validation. This is used to ensure
                          the file can be loaded at the desired sample rate.
    """
    try:
        if not os.path.exists(audio_path):
            logging.warning(f"Audio file not found: {audio_path}")
            return False
        # Try loading a small portion to validate at the target sample rate
        librosa.load(audio_path, sr=sample_rate, duration=0.1)
        return True
    except Exception as e:
        logging.warning(f"Error loading audio file {audio_path}: {str(e)}")
        return False

def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    normalize: bool = True,
    trim_silence: bool = False,
    target_length: Optional[int] = None
) -> np.ndarray:
    """
    Apply preprocessing steps to audio data.
    
    Args:
        audio (np.ndarray): Raw audio data
        sample_rate (int): Sample rate of the audio. This is used for silence trimming
                          and to calculate target length in seconds if needed.
        normalize (bool): Whether to normalize audio data
        trim_silence (bool): Whether to trim silence from audio
        target_length (Optional[int]): Target length in samples. If None, no length adjustment is done.
                                      If specified, audio will be padded or truncated to this length.
    """
    if normalize:
        audio = librosa.util.normalize(audio)
    
    if trim_silence:
        # trim_silence uses the sample rate to determine what constitutes silence
        audio, _ = librosa.effects.trim(audio, top_db=30)
    
    if target_length is not None:
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))
    
    return audio

def load_anuraset_data(
    base_path: str = "../anuraset/anuraset",
    subset: str = None,
    sample_rate: int = 22050,
    batch_size: Optional[int] = None,
    normalize: bool = True,
    trim_silence: bool = False,
    target_length: Optional[int] = None,
    validation_split: float = 0.2,
    random_seed: int = 42
) -> Union[Tuple[pd.DataFrame, Dict[str, np.ndarray]], List[Tuple[pd.DataFrame, Dict[str, np.ndarray]]]]:
    """
    Load the Anuraset dataset including metadata and audio files with enhanced features.
    
    Args:
        base_path (str): Path to the Anuraset dataset directory
        subset (str, optional): Filter by subset ('train', 'test', or None for all)
        sample_rate (int): Target sample rate for audio loading. This is crucial for:
                          - Loading audio files at a consistent rate
                          - Ensuring all audio samples have the same temporal resolution
                          - Proper functioning of audio processing operations
                          Default is 22050 Hz (half of CD quality), which is common for
                          audio processing tasks and provides good quality while being
                          computationally efficient.
        batch_size (int, optional): If provided, returns data in batches of specified size
        normalize (bool): Whether to normalize audio data
        trim_silence (bool): Whether to trim silence from audio
        target_length (int, optional): Target length for audio samples in samples. If None,
                                     original lengths are preserved. If specified, all audio
                                     will be padded or truncated to this length.
        validation_split (float): Fraction of data to use for validation (if batch_size is None)
        random_seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load metadata
    metadata_path = os.path.join(base_path, "filtered_metadata2.csv")
    metadata = pd.read_csv(metadata_path)
    
    # Filter by subset if specified
    if subset:
        metadata = metadata[metadata['subset'] == subset]
    
    # Shuffle data
    metadata = metadata.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split into train/validation if not using batches
    if batch_size is None and validation_split > 0:
        split_idx = int(len(metadata) * (1 - validation_split))
        train_metadata = metadata[:split_idx]
        val_metadata = metadata[split_idx:]
        metadata_splits = [train_metadata, val_metadata]
    else:
        metadata_splits = [metadata]
    
    results = []
    
    for current_metadata in metadata_splits:
        # Initialize audio dictionary
        audio_data = {}
        missing_files = []
        
        # Process audio files with progress bar
        for _, row in tqdm(current_metadata.iterrows(), total=len(current_metadata), desc="Loading audio files"):
            wav_path = f'{base_path}/audio/{row["site"]}/{row["fname"]}_{row["min_t"]}_{row["max_t"]}.wav'
            
            if validate_audio_file(wav_path, sample_rate):
                try:
                    # Load audio at the specified sample rate
                    audio, actual_sr = librosa.load(wav_path, sr=sample_rate)
                    
                    # Verify the sample rate matches what we expect
                    if actual_sr != sample_rate:
                        logging.warning(f"Sample rate mismatch in {wav_path}: expected {sample_rate}, got {actual_sr}")
                    
                    audio = preprocess_audio(
                        audio,
                        sample_rate,  # Pass sample rate to preprocessing
                        normalize=normalize,
                        trim_silence=trim_silence,
                        target_length=target_length
                    )
                    audio_data[row['sample_name']] = audio
                except Exception as e:
                    missing_files.append((wav_path, str(e)))
            else:
                missing_files.append((wav_path, "File not found or invalid"))
        
        if missing_files:
            logging.warning(f"Failed to load {len(missing_files)} files. First few errors:")
            for path, error in missing_files[:5]:
                logging.warning(f"{path}: {error}")
        
        if batch_size is not None:
            # Split into batches
            num_batches = (len(current_metadata) + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(current_metadata))
                batch_metadata = current_metadata.iloc[start_idx:end_idx]
                batch_audio = {name: audio_data[name] for name in batch_metadata['sample_name'] 
                             if name in audio_data}
                results.append((batch_metadata, batch_audio))
        else:
            results.append((current_metadata, audio_data))
    
    return results[0] if len(results) == 1 else results

def plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Audio Waveform",
    figsize: Tuple[int, int] = (12, 4),
    color: str = 'b',
    alpha: float = 0.6,
    show_grid: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Plot an audio waveform with proper time axis and scaling.
    
    Args:
        audio (np.ndarray): Audio data to plot
        sample_rate (int): Sample rate of the audio
        title (str): Title for the plot
        figsize (Tuple[int, int]): Size of the figure (width, height)
        color (str): Color of the waveform
        alpha (float): Transparency of the waveform
        show_grid (bool): Whether to show grid lines
        save_path (Optional[str]): If provided, save the plot to this path
    """
    # Create time axis
    time = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot waveform
    plt.plot(time, audio, color=color, alpha=alpha)
    
    # Customize plot
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(show_grid)
    
    # Set y-axis limits to show full range
    plt.ylim(-1.1, 1.1)
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Basic usage
    metadata, audio_data = load_anuraset_data()
    
    # With preprocessing
    metadata, audio_data = load_anuraset_data(
        normalize=True,
        trim_silence=True,
        target_length=22050  # 1 second at 22.05kHz
    )
    
    # With batching
    batches = load_anuraset_data(
        batch_size=32,
        normalize=True
    )
    
    # With validation split
    train_metadata, train_audio = load_anuraset_data(
        subset="train",
        validation_split=0.2
    )

    # Plot a few examples
    if isinstance(audio_data, dict):  # Check if we have a single dataset
        for sample_name, audio in list(audio_data.items())[:3]:
            plot_waveform(
                audio,
                sample_rate=22050,
                title=f"Waveform: {sample_name}",
                save_path=f"waveform_{sample_name}.png"
            )
    else:  # We have batches
        for batch_idx, (batch_metadata, batch_audio) in enumerate(batches[:1]):  # Just plot first batch
            for sample_name, audio in list(batch_audio.items())[:3]:
                plot_waveform(
                    audio,
                    sample_rate=22050,
                    title=f"Waveform: {sample_name} (Batch {batch_idx})",
                    save_path=f"waveform_batch{batch_idx}_{sample_name}.png"
                )

  # 1. Validate Your Data Pipeline by plooting a few audio waveforms and spectrograms
