from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE


def get_all_vectors_from_faiss_index(index):
    num_vectors = index.ntotal
    all_vectors = np.empty((num_vectors, index.d), dtype=np.float32)
    batch_size = 1000
    for start in range(0, num_vectors, batch_size):
        end = min(start + batch_size, num_vectors)
        vectors_batch = index.reconstruct_n(start, end - start)
        all_vectors[start:end, :] = vectors_batch
    return all_vectors
