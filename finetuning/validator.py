#!/usr/bin/env python3
"""
forwardpass.py

This script demonstrates a simple forward pass through your audio transformer model.
It loads a pretrained model checkpoint (e.g., animal2vec or one-peace), creates a dummy
audio input (simulating a 10-second audio segment at an 8000 Hz sample rate), processes
the input, and prints the output shapes of the model's predictions.

Before running, update the `model_path` variable to point to your checkpoint file.
Make sure you run this from the repository’s root directory so that all modules (like `nn`)
are available.
"""

import torch
import numpy as np
from fairseq import checkpoint_utils
import nn  # Assumes your repo exposes functions like chunk_and_normalize

def main():
    # Update this path to your model checkpoint.
    model_path = "./animal2vec_large_pretrained_MeerKAT_240507.pt"  # e.g., "checkpoints/animal2vec_checkpoint.pt"
    print("Loading model from {}...".format(model_path))
    # Load the model checkpoint using fairseq utility.
    models, model_args = checkpoint_utils.load_model_ensemble([model_path])
    model = models[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Model loaded and moved to {}.".format(device))

    # -----------------------------------------------------------------------------
    # Create dummy audio input:
    # For a 10-second clip sampled at 8000 Hz, we need 10 * 8000 = 80000 samples.
    # The expected input shape is Batch x Time, so here we use a batch size of 1.
    # -----------------------------------------------------------------------------
    dummy_data = torch.rand(1, 80000)  # Randomly generated audio waveform
    print("Dummy data shape before processing:", dummy_data.shape)

    # -----------------------------------------------------------------------------
    # Process the input using the provided normalization and chunking helper function.
    # This function is expected to:
    #   - Normalize the audio (if needed)
    #   - Chunk the audio into segments (here, only one segment given our size)
    # Update parameters as needed (segment_length is in seconds, sample_rate in Hz).
    # -----------------------------------------------------------------------------
    dummy_data_chunks = nn.chunk_and_normalize(
        dummy_data,
        segment_length=10,    # Duration in seconds (matches our dummy data)
        sample_rate=8000,     # Sample rate of the data
        normalize=True,       # Whether to normalize to zero mean/unit variance
        max_batch_size=16     # Maximum batch size for internal processing
    )
    
    print("Number of chunks produced:", len(dummy_data_chunks))
    
    # -----------------------------------------------------------------------------
    # Run a forward pass on each processed chunk.
    # We use torch.inference_mode() to disable gradient computation during this test.
    # The model output is expected to be a dictionary containing (for example) 'encoder_out',
    # which holds the logits that can later be turned into probabilities.
    # -----------------------------------------------------------------------------
    print("Running forward pass on processed chunks:")
    with torch.inference_mode():
        for i, chunk in enumerate(dummy_data_chunks):
            # Ensure that the chunk is moved to the correct device.
            chunk = chunk.to(device)
            net_output = model(source=chunk)

            # Check if the expected key ("encoder_out") is present in the model output.
            if "encoder_out" in net_output:
                # Convert logits to probabilities using the sigmoid function.
                probs = torch.sigmoid(net_output["encoder_out"].float())
                print(f"Chunk {i}: output shape {probs.shape}")
            else:
                print(f"Warning: 'encoder_out' key not found in output for chunk {i}")

if __name__ == "__main__":
    main()
