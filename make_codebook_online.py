"""
Build the ONLINE (768-D SED+HR) language codebook for loop closure.

Adapted from the original make_language_codebook.py (which built the 512-D LSeg
codebook). Methodology is preserved exactly — MiniBatchKMeans(64), incremental
partial_fit, 5% per-image pixel sampling, ScanNet source, seed 42 — the ONLY
change is the feature source: instead of loading precomputed 512-D LSeg .pt files,
we extract 768-D SED+HR features live from ScanNet color images (same pipeline as
online SLAM), so the codebook lives in the same space as online features.

Output: saved/language_codebook_online.pkl  (vocabulary: 64 x 768)
"""
import os
import glob
import random
import time
import pickle
import argparse

import cv2
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from clip_sed.extractor import extract  # SED+HR -> (768, 192, 192), identical to online

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scannet_root", required=True, help="Path to ScanNet root (scene*/color/*.jpg)")
SCANNET_ROOT = parser.parse_args().scannet_root
NUM_CLUSTERS = 64
TARGET_VECTORS = 3_000_000          # reduced vs original 100M: live extraction is slower, ample for k=64
USING_RATIO = 0.05                  # 5% of pixels per image (same as original)
OUT_PATH = "saved/language_codebook_online.pkl"


def main():
    random.seed(42)
    print(f"[codebook-online] SED+HR 768-D, k={NUM_CLUSTERS}, target={TARGET_VECTORS:,}, ratio={USING_RATIO}")

    scenes = sorted(d for d in os.listdir(SCANNET_ROOT)
                    if d.startswith("scene") and os.path.isdir(os.path.join(SCANNET_ROOT, d)))
    pixels_per_img = 192 * 192                       # SED+HR output resolution
    sampled_per_img = int(pixels_per_img * USING_RATIO)
    files_needed = TARGET_VECTORS // sampled_per_img + 1
    per_scene = files_needed // max(len(scenes), 1) + 1
    print(f"[codebook-online] scenes={len(scenes)}, files_needed~{files_needed}, per_scene~{per_scene}")

    sampled = []
    for sc in scenes:
        imgs = sorted(glob.glob(os.path.join(SCANNET_ROOT, sc, "color", "*.jpg")))
        if not imgs:
            continue
        sampled += random.sample(imgs, min(per_scene, len(imgs)))
    random.shuffle(sampled)
    print(f"[codebook-online] sampled images: {len(sampled)}")

    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, random_state=42,
                             batch_size=10000, max_iter=100, n_init=3, verbose=0)

    total = 0
    t0 = time.time()
    for i, img_path in enumerate(tqdm(sampled, desc="extract+fit")):
        try:
            img = cv2.imread(img_path)            # BGR — same as online SLAM feeds to extract()
            feat = extract(img).float()           # (768, 192, 192)
            C, H, W = feat.shape
            vecs = feat.permute(1, 2, 0).reshape(-1, C)
            n = int(vecs.shape[0] * USING_RATIO)
            idx = torch.randperm(vecs.shape[0])[:n]
            vecs_np = vecs[idx].numpy()
            kmeans.partial_fit(vecs_np)
            total += vecs_np.shape[0]
            del feat, vecs, vecs_np
            if total >= TARGET_VECTORS:
                print(f"\n[codebook-online] reached target: {total:,}")
                break
        except Exception as e:
            print(f"[codebook-online] skip {os.path.basename(img_path)}: {e}")
            continue

    vocab = kmeans.cluster_centers_  # (64, 768)
    os.makedirs("saved", exist_ok=True)
    pickle.dump({
        'vocabulary': vocab,
        'kmeans_model': kmeans,
        'num_clusters': NUM_CLUSTERS,
        'vector_dimension': 768,
        'total_vectors_used': total,
        'target_vectors': TARGET_VECTORS,
        'sampled_files': sampled[:i + 1],
        'inertia': float(kmeans.inertia_),
        'creation_time': time.time(),
        'using_ratio': USING_RATIO,
        'source': 'SED+HR live extraction (ScanNet color)',
    }, open(OUT_PATH, 'wb'))
    print(f"[codebook-online] saved {OUT_PATH}  vocab={vocab.shape}  vectors={total:,}  "
          f"inertia={kmeans.inertia_:.2e}  time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
