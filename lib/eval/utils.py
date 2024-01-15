from collections import defaultdict

import numpy as np


def get_all_vectors_from_faiss_index(index):
    num_vectors = index.ntotal
    all_vectors = np.empty((num_vectors, index.d), dtype=np.float32)
    batch_size = 1000
    for start in range(0, num_vectors, batch_size):
        end = min(start + batch_size, num_vectors)
        vectors_batch = index.reconstruct_n(start, end - start)
        all_vectors[start:end, :] = vectors_batch
    return all_vectors


def batch_outputs(data):
    out = defaultdict(list)
    for outputs in data:
        for key, value in outputs.items():
            out[key].append(value)
    for key, value in out.items():
        out[key] = np.concatenate(value)  # type: ignore
    return out


def detach_batch_output(batch, output):
    return {
        "sketch_emb": output["sketch_emb"].detach().cpu().numpy(),
        "image_emb": output["image_emb"].detach().cpu().numpy(),
        "label": batch["label"].detach().cpu().numpy(),
        "sketch": batch["sketch"].detach().cpu().numpy(),
        "image": batch["image"].detach().cpu().numpy(),
    }
