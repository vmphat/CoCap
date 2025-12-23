import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
from pathlib import Path
from torchvision import io
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from typing import List, Dict
import os
os.environ["TORCH_HOME"] = "/media02/lnthanh01/vmphat/raw_data/cache"


# --- Constants for loading ---
KEYINT: int = 40
CORPUS: str = "vatex"
RESIZE: int = 240
EXTENSION: str = "mp4"
TOTAL_VIDEOS: int = 34_793
# --- Constants for processing ---
CHUNK_SIZE: int = 16
STACK_SIZE: int = KEYINT
STEP_SIZE : int = KEYINT


# --- Helper functions ---
def resample_fixed(frames, chunk_size):
    C, T, H, W = frames.shape
    idxs = np.linspace(0, T-1, chunk_size).astype(int)
    return frames[:, idxs, :, :]


def chunk_frames(inputs, chunk_size, stack_size, step_size):
    """ Chunk input frames into overlapping chunks.

    Args:
        inputs: torch.Tensor of shape (C, T, H, W)
        chunk_size: int, number of frames per chunk after resampling
        stack_size: int, number of frames to stack before resampling
        step_size: int, step size for moving the window

    Returns:
        List of torch.Tensor, each of shape (C, chunk_size, H, W)
    """
    C, T, H, W = inputs.shape
    chunked_inputs = []

    for i in range(0, T, step_size):
        start_idx = i
        end_idx = min(i + stack_size, T)
        num_frames = end_idx - start_idx
        # print(f"Chunk {i}, from {start_idx} to {end_idx} => num_frames={num_frames}")

        if num_frames < stack_size:  # last batch
            chunk = inputs[:, -(stack_size):, :, :]
        else:
            chunk = inputs[:, start_idx:end_idx, :, :]

        # Resample chunk to have exactly chunk_size frames
        chunk = resample_fixed(chunk, chunk_size=chunk_size)
        chunked_inputs.append(chunk)

    # Sanity check
    if len(chunked_inputs) == 0:
        raise ValueError("No chunks created from input frames")

    num_gop_target = inputs[:, ::KEYINT, :, :].shape[1] + 1
    # ***** Add last chunk for the last frame *****
    if len(chunked_inputs) < num_gop_target:
        last_chunk = inputs[:, -(stack_size):, :, :]
        last_chunk = resample_fixed(last_chunk, chunk_size=chunk_size)
        chunked_inputs.append(last_chunk)
    # *********************************************
    assert len(chunked_inputs) == num_gop_target, f"{len(chunked_inputs)} vs {num_gop_target}"

    return chunked_inputs


def extract_features_for_video(video_path,
                               preprocess, chunk_size, stack_size, step_size,
                               model, device, batch_size):
    """Extract features from a video file.

    Args:
        video_path: str, path to video file
        preprocess: preprocessing function
        chunk_size: int, number of frames per chunk after resampling
        stack_size: int, number of frames to stack before resampling
        step_size: int, step size for moving the window
        model: feature extraction model
        device: torch device
        batch_size: int, batch size for processing chunks

    Returns:
        np.ndarray of shape (N_chunks, feature_dim)
    """

    # --- load video ---
    video_data, _, info = io.read_video(
        filename=video_path,
        pts_unit='sec',
        output_format="TCHW"
    )
    assert video_data.ndim == 4  # T, C, H, W
    assert video_data.shape[1] == 3  # C=3

    # --- preprocess ---
    inputs = preprocess(video_data)  # C, T, H, W
    assert inputs.ndim == 4
    assert inputs.shape[0] == 3  # C=3

    # --- chunk frames ---
    chunked_inputs = chunk_frames(
        inputs,
        chunk_size=chunk_size,
        stack_size=stack_size,
        step_size=step_size
    )

    # --- extract features in batches ---
    all_feats = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(chunked_inputs), batch_size):
            batch_chunks = chunked_inputs[i:i+batch_size]
            # (B, C, chunk_size, H, W)
            batch_tensor = torch.stack(batch_chunks).to(device)

            # inference
            outputs = model(batch_tensor)  # (B, feature_dim)
            outputs = outputs.cpu().numpy().astype(np.float32)

            all_feats.append(outputs)

    if len(all_feats) == 0:
        raise ValueError("No features extracted from video")

    feats = np.vstack(all_feats)  # shape (N_chunks, feature_dim)
    return feats


# --- Get path to all videos ---
print("Loading video paths...")
video_paths: List[str] = list(Path(f"./{CORPUS.lower()}/videos_{RESIZE}_h264_keyint_{KEYINT}/").glob(f"*.{EXTENSION}"))
print(f">> Found {len(video_paths)} videos.")
assert len(video_paths) == TOTAL_VIDEOS, f"Expected {TOTAL_VIDEOS} videos, but found {len(video_paths)}."

# --- load pretrained model + transforms ---
print(f"Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = MViT_V2_S_Weights.DEFAULT
preprocess = weights.transforms()
model = mvit_v2_s(weights=weights).to(device)
# remove head + norm to get feature vectors
model.head = torch.nn.Identity()
model.norm = torch.nn.Identity()
model.eval()

# --- extract features for all videos ---
vid_to_gop_count: Dict[str, int] = {}
mot_emb_hdf5 = h5py.File(f"./{CORPUS.upper()}_newMViTv2.hdf5", "a")

print(f"Extracting feature...")
for video_path in tqdm(video_paths, desc="Videos"):
    if video_path.stem not in mot_emb_hdf5:
        # extract features
        feats = extract_features_for_video(
            video_path=str(video_path),
            preprocess=preprocess,
            chunk_size=CHUNK_SIZE,
            stack_size=STACK_SIZE,
            step_size=STEP_SIZE,
            model=model,
            device=device,
            batch_size=8
        )  # (N_chunks, feature_dim)

        # save feature to HDF5
        mot_emb_hdf5.create_dataset(video_path.stem, data=feats)
        # save gop count
        vid_to_gop_count[video_path.stem] = feats.shape[0]

# Close HDF5 file
mot_emb_hdf5.close()

# Save GOP count to CSV
df = pd.DataFrame(list(vid_to_gop_count.items()),
                  columns=["video_id", "gop_count"])
df.to_csv(f"./{CORPUS.lower()}_new_gop_counts.csv", index=False)

print(">> Done.")
