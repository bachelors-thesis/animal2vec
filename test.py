# The 'nn' import only works if you are in the root directory of the a2v repo, 
# This is always needed to register the model and tasks objects. Otherwise, the
# fairseq routines will through an error using our models
import nn
import torch
import numpy as np
from fairseq import checkpoint_utils
import tensorflow as tf

# List all physical GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

# These are the class names in the MeerKAT dataset
meerkat_class_names = ['beep', 'synch', 'sn', 'cc', 'ld', 'oth', 'mo', 'al', 'soc', 'agg', 'eating', 'focal']
path_to_pt_file = "animal2vec_large_finetuned_MeerKAT_240507.pt"
# load the model
print("\n Loading model ... ", end="")
models, model_args = checkpoint_utils.load_model_ensemble([path_to_pt_file])
print("done")

print("Moving model to cpu ... ", end="")
model = models[0].to("cpu")  # place on appropriate device
print("done\n")

# Expected shape is Batch x Time. This simulates a 10s segment at 8kHz sample rate
dummy_data = torch.rand(1, 80000).cpu()

# Generally, you should always normalize your input to zero mean and unit variance
# This repository has a helper function for that
dummy_data = nn.chunk_and_normalize(
    dummy_data,
    segment_length=10,  # Length in seconds
    sample_rate=8000,  # Sample rate of your data
    normalize=True,
    max_batch_size=16  # The max batch size your GPU or RAM can handle (16 should be ok)
)

processed_chunks = []
method_dict = {
        "sigma_s": 0.1,  # Filter width in seconds
        "metric_threshold": 0.15,  # Likelihood threshold for the predictions
        "maxfilt_s": 0.1,  # Time to smooth with maxixmum filter. Only used when method=canny
        "max_duration_s": 0.5,  # Detections are never longer than max duration (s). Only used when method=canny
        "lowP": 0.125,  # Lower likelihood threshold. Only used when method=canny
    }
with torch.inference_mode():
    model.eval()
    for bi, single_chunk in enumerate(dummy_data):
        # lists should be stacked and 1-dimensional data should be extended to 2d.
        if not torch.is_tensor(single_chunk):
            single_chunk = torch.stack(single_chunk)  # stack the list of tensors
        elif single_chunk.dim() == 1:  # split segments or single segment
            single_chunk = single_chunk.view(1, -1)
        # 1) Get frame_wise predictions
        
        # This returns a dictionary with keys: ['encoder_out', 'padding_mask', 'layer_results', 'target']
        # encoder_out is the classifier logits output (use torch.sigmoid to turn into probs)
        # padding_mask is the used padding mask (usually no padding is used, then, padding_mask is None)
        # layer_results is a list that holds the embeddings from all transformer layers
        # target is the ground truth (if provided, usually this is None, as we are not training anymore)
        net_output = model(source=single_chunk.to("cpu"))
        
        # 1.1) Convert to probalities. This has shape Batch x Time x Class (1, 2000, 12 in this example)
        probs = torch.sigmoid(net_output["encoder_out"].float())
        
        # 2) Get onset / offset predictions
        # This function calculates onset and offset and the average likelihood between the found
        # boundaries. It returns a list with len 3 with onset/offset info in seconds, their indexes,
        # and the likelihood for that segment, for every class
        fused_preds = model.fuse_predict(single_chunk.size(-1), probs,
                                        # A dictionary with information on how to estimate the onset / offset
                                         method_dict=method_dict,
                                         # Which method to use for fusing the predictions into time bins
                                         method="avg",
                                         multiplier=bi,
                                         bs=16)
        processed_chunks.append(fused_preds)

        print("We iterate over {} chunks".format(len(processed_chunks)))
        for ci, single_chunk in enumerate(processed_chunks):  # iterate over all chunks
            time_interval_batches = single_chunk[0]  # time in seconds
            likelihoods_batches = single_chunk[2]  # likelihood between 0 and 1
            
            # iterate over the segments in each chunk 
            print("\tChunk {} has {} segments".format(ci, len(time_interval_batches)))
            for t_batch, l_batch in zip(time_interval_batches, likelihoods_batches):
                # iterate over the class predictions in each batch in each chunk 
                for si, (t_seg, l_seg, n_) in enumerate(zip(t_batch, l_batch, meerkat_class_names)):
                    print("\t\tResults for Class {}: {}".format(si, n_))
                    print("\t\t\tClass {} has {} found segments.".format(n_, len(t_seg)))
                    for class_pred_time, class_pred_like in zip(t_seg, l_seg):
                        pr_args = (class_pred_time[0].numpy(), class_pred_time[1].numpy(), class_pred_like.numpy())
                        print("\t\t\t\tFrom {:02.02f}s to {:02.02f}s with a likelihood of {:02.02f}".format(*pr_args))

# With this simple example, we get an output of this structure:
#   We iterate over 1 chunks
#   	Chunk 0 has 1 segments
#   		Results for Class 0: beep
#   			Class beep has 0 found segments.
#   		Results for Class 1: synch
#   			Class synch has 1 found segments.
#   				From 0.05s to 0.32s with a likelihood of 0.24
#   		Results for Class 2: sn
#   			Class sn has 6 found segments.
#   				From 0.05s to 0.07s with a likelihood of 0.14
#   				From 0.11s to 0.19s with a likelihood of 0.16
#   				From 2.00s to 2.09s with a likelihood of 0.19
#   				From 2.48s to 2.55s with a likelihood of 0.18
#   				From 2.73s to 2.80s with a likelihood of 0.18
#   				From 4.07s to 4.16s with a likelihood of 0.19
#   		Results for Class 3: cc
#   			Class cc has 0 found segments.
#   		Results for Class 4: ld
#   			Class ld has 0 found segments.
#   		Results for Class 5: oth
#   			Class oth has 0 found segments.
#   		Results for Class 6: mo
#   			Class mo has 0 found segments.
#   		Results for Class 7: al
#   			Class al has 0 found segments.
#   		Results for Class 8: soc
#   			Class soc has 0 found segments.
#   		Results for Class 9: agg
#   			Class agg has 0 found segments.
#   		Results for Class 10: eating
#   			Class eating has 0 found segments.
#   		Results for Class 11: focal
#   			Class focal has 1 found segments.
#   				From 0.05s to 0.29s with a likelihood of 0.23

# Please note, the actual found segments will vary, as the input was created using random numbers.