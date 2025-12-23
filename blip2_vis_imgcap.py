import cv2
import json
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# --- Constants for loading ---
KEYINT: int = 40
CORPUS: str = "vatex"
CACHE_DIR: str = "/media02/lnthanh01/vmphat/raw_data/cache"
RESIZE: int = 240
EXTENSION: str = "mp4"
TOTAL_VIDEOS: int = 34_793


# --- Helper functions ---
def extract_frames_as_pil(video_path: str) -> List[Image.Image]:
    """ Extract all frames from a video and return them as a list of PIL.Image objects.

    Parameters
    - video_path: path to input video file.
    
    Returns
    - List of PIL.Image.Image objects (RGB)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frames: List[Image.Image] = []

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            # convert BGR (OpenCV) to RGB (PIL)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)
    finally:
        cap.release()

    return frames


def extract_visual_embeddings(frames_pil, processor, vision_model, device, batch_size):
    """
    Args:
        frames_pil: list of PIL.Image
        processor: Blip2Processor
        vision_model: Blip2ForConditionalGeneration.vision_model
    """

    all_cls_tokens = []
    all_avg_tokens = []
    vision_model.to(device)
    vision_model.eval()

    with torch.no_grad():
        for i in range(0, len(frames_pil), batch_size):
            # Get batch of images
            batch_imgs = frames_pil[i:i+batch_size]

            # Processor will resize/normalize according to model's default transforms
            inputs = processor(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            # Inference
            outputs = vision_model(pixel_values=pixel_values)

            # === Extract CLS token ===
            cls_token = outputs.pooler_output  # (B, hidden_dim)
            cls_token = cls_token.cpu().numpy().astype(np.float32)
            all_cls_tokens.append(cls_token)

            # === Extract Average Pooling token ===
            # outputs.last_hidden_state: (B, seq_len, hidden_dim)
            avg_token = outputs.last_hidden_state.mean(dim=1)  # (B, hidden_dim)
            avg_token = avg_token.cpu().numpy().astype(np.float32)
            all_avg_tokens.append(avg_token)

    if (len(all_cls_tokens) == 0) or (len(all_avg_tokens) == 0):
        raise ValueError("No features extracted")

    vis_emb_cls = np.vstack(all_cls_tokens)  # shape (N_frames, hidden_dim)
    vis_emb_avg = np.vstack(all_avg_tokens)  # shape (N_frames, hidden_dim)
    return vis_emb_cls, vis_emb_avg


# --- Full pipeline: captions to embeddings ---
def captions_to_embeddings_pipeline(
    image_inputs: List[Image.Image],
    blip_processor: Blip2Processor,
    blip_model: Blip2ForConditionalGeneration,
    embed_model: SentenceTransformer,
    device: str,
    batch_size: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Process images and generate embeddings using BLIP and SentenceTransformer.

    Args:
        image_inputs: List of PIL.Image objects.
        blip_processor: BLIP processor for image preprocessing.
        blip_model: BLIP model for image captioning.
        embed_model: SentenceTransformer model for text embedding.
        device: Device to run the model on ('cpu' or 'cuda').
        batch_size: Batch size for processing images.
    Returns:
        - Numpy array of embeddings with shape (N, embedding_dim).
        - A list of captions for each (key-)frame.
    """

    # Initialize models
    blip_model.to(device)
    blip_model.eval()
    embed_model.to(device)
    embed_model.eval()

    all_embeddings: List[np.ndarray] = []
    all_captions: List[str] = []
    with torch.no_grad():
        for i in range(0, len(image_inputs), batch_size):
            batch_images: List[Image.Image] = image_inputs[i:i+batch_size]

            # Preprocess images
            inputs = blip_processor(images=batch_images, return_tensors="pt").to(device)

            # Generate captions
            outputs = blip_model.generate(**inputs, max_new_tokens=32, num_beams=4)
            captions = blip_processor.batch_decode(outputs, skip_special_tokens=True)
            all_captions += captions

            # Generate embeddings
            embeddings = embed_model.encode(captions)
            all_embeddings.append(embeddings)

    if len(all_embeddings) == 0:
        raise ValueError("No embeddings extracted")

    all_embeddings: np.ndarray = np.vstack(all_embeddings)  # shape (N, embedding_dim)
    
    return all_embeddings, all_captions


# --- Get path to all videos ---
print("Loading video paths...")
video_paths: List[str] = list(Path(f"./{CORPUS.lower()}/videos_{RESIZE}_h264_keyint_{KEYINT}/").glob(f"*.{EXTENSION}"))
print(f">> Found {len(video_paths)} videos.")
assert len(video_paths) == TOTAL_VIDEOS, f"Expected {TOTAL_VIDEOS} videos, but found {len(video_paths)}."

# --- Load processor and models ---
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b", cache_dir=CACHE_DIR)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", cache_dir=CACHE_DIR).to(device)
embed_model = SentenceTransformer(
    "all-roberta-large-v1", device=device, cache_folder=CACHE_DIR)
blip_vision_model = blip_model.vision_model  # encoder for images

# --- Extract keyframes, image embeddings, embeddings from image captions ---
vid_to_keyframe_captions: List[Dict] = []
vid_to_keyframe_count: Dict[str, int] = {}
vis_emb_cls_hdf5 = h5py.File(f"./{CORPUS.upper()}_newBlip2ClsKF.hdf5", "a")
vis_emb_avg_hdf5 = h5py.File(f"./{CORPUS.upper()}_newBlip2AvgKF.hdf5", "a")
cap_emb_hdf5     = h5py.File(f"./{CORPUS.upper()}_newImgCapBlip2KF.hdf5", "a")

print("Extracting features for each video...")
for video_path in tqdm(video_paths, desc="Processing videos"):
    # === Extract keyframes ===
    all_frames = extract_frames_as_pil(str(video_path))
    keyframes = all_frames[::KEYINT]    # Take every KEYINT-th frame
    keyframes.append(all_frames[-1])    # Ensure the last frame is included
    vid_to_keyframe_count[video_path.stem] = len(keyframes)
    assert len(keyframes) == (len(all_frames[::KEYINT]) + 1)

    # === Extract visual embeddings (CLS and AVG) ===
    if (video_path.stem not in vis_emb_cls_hdf5) or (video_path.stem not in vis_emb_avg_hdf5):    
        vis_emb_cls, vis_emb_avg = extract_visual_embeddings(
            frames_pil=keyframes,
            processor=blip_processor,
            vision_model=blip_vision_model,
            device=device,
            batch_size=8
        )
        if video_path.stem not in vis_emb_cls_hdf5:
            vis_emb_cls_hdf5.create_dataset(video_path.stem, data=vis_emb_cls)
        if video_path.stem not in vis_emb_avg_hdf5:
            vis_emb_avg_hdf5.create_dataset(video_path.stem, data=vis_emb_avg)

    # === Extract caption embeddings from images ===
    if video_path.stem not in cap_emb_hdf5:
        caption_embs, caption_list = captions_to_embeddings_pipeline(
            image_inputs=keyframes,
            blip_processor=blip_processor,
            blip_model=blip_model,
            embed_model=embed_model,
            device=device,
            batch_size=8
        )
        cap_emb_hdf5.create_dataset(video_path.stem, data=caption_embs)
        vid_to_keyframe_captions.append({
            "videoID": video_path.stem,
            "keyframeCap": caption_list
        })
        print(f"-"*46)
        print(f"{video_path.stem}:\n{caption_list}")
        print(f"="*46)

# Close HDF5 files
vis_emb_cls_hdf5.close()
vis_emb_avg_hdf5.close()
cap_emb_hdf5.close()

# Save keyframe counts to CSV
df = pd.DataFrame(list(vid_to_keyframe_count.items()),
                  columns=["video_id", "keyframe_count"])
df.to_csv(f"./{CORPUS.lower()}_new_keyframe_counts.csv", index=False)
# Save caption-list to JSON
with open(f"./{CORPUS.lower()}_new_keyframe_captions.json", "w") as f:
    json.dump(vid_to_keyframe_captions, f, indent=4)

print(">> Done!")
