"""Metrics for evaluating SignViP token sequences in continuous FSQ space."""

import numpy as np
import torch
import torch.nn.functional as F

from models.condition_encoder import VQModel


def load_fsq_decoder(checkpoint, model_kwargs, device, dtype):
    """Load only the FSQ autoencoder used to decode token grids."""
    model = VQModel(**model_kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prefixes = ("module.vq.", "vq.")
    vq_state = {
        name[len(prefix) :]: value
        for name, value in state.items()
        for prefix in prefixes
        if name.startswith(prefix)
    }
    if not vq_state:
        vq_state = state
    model.load_state_dict(vq_state, strict=True)
    model.requires_grad_(False).eval()
    return model.to(device=device, dtype=dtype)


def ground_truth_frames(flat_tokens, token_length, frame_codes, codebook_size):
    """Remove padding and the final all-EOS frame from a dataset target."""
    token_length = int(token_length)
    usable = token_length - frame_codes
    if usable <= 0 or usable % frame_codes:
        raise ValueError(
            f"Invalid target length {token_length} for {frame_codes} codes per frame."
        )
    frames = flat_tokens[:usable].reshape(-1, frame_codes)
    if torch.any(frames >= codebook_size):
        raise ValueError("Ground-truth content contains EOS/PAD token IDs.")
    return frames.long()


def predicted_frames(flat_tokens, frame_codes, codebook_size):
    """Apply the same first-EOS frame truncation used by video inference."""
    flat_tokens = flat_tokens.flatten()
    stop_positions = torch.nonzero(flat_tokens >= codebook_size, as_tuple=False)
    stop = int(stop_positions[0]) if len(stop_positions) else flat_tokens.numel()
    usable = stop - stop % frame_codes
    if usable == 0:
        return flat_tokens.new_empty((0, frame_codes), dtype=torch.long)
    return flat_tokens[:usable].reshape(-1, frame_codes).long()


@torch.inference_mode()
def decode_embedding_sequence(decoder, tokens, pool_size=(4, 4), chunk_size=32):
    """Decode FSQ grids and retain a compact spatial map for DTW."""
    outputs = []
    for start in range(0, len(tokens), chunk_size):
        decoded = decoder.decode(tokens[start : start + chunk_size])
        if pool_size:
            decoded = F.adaptive_avg_pool2d(decoded, tuple(pool_size))
        outputs.append(decoded.float().flatten(1))
    return torch.cat(outputs, dim=0)


def pairwise_rmse(first, second):
    """Pairwise root-mean-square distance without materializing feature deltas."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("Expected two [time, feature] tensors with matching features.")
    dimensions = first.shape[1]
    first_norm = first.square().sum(1, keepdim=True)
    second_norm = second.square().sum(1).unsqueeze(0)
    squared = first_norm + second_norm - 2 * (first @ second.transpose(0, 1))
    return (squared.clamp_min_(0) / dimensions).sqrt_()


def exact_dtw(distance_matrix):
    """Return exact DTW cost for a precomputed local-distance matrix."""
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or 0 in distances.shape:
        raise ValueError("DTW requires a non-empty two-dimensional distance matrix.")
    previous = np.full(distances.shape[1] + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for row in distances:
        current = np.full_like(previous, np.inf)
        for column, cost in enumerate(row, start=1):
            current[column] = cost + min(
                current[column - 1], previous[column], previous[column - 1]
            )
        previous = current
    return float(previous[-1])


@torch.inference_mode()
def normalized_decoded_dtw(
    decoder,
    predicted_tokens,
    target_tokens,
    pool_size=(4, 4),
    chunk_size=32,
):
    """Compute normalized DTW between decoded continuous condition embeddings."""
    if len(predicted_tokens) == 0 or len(target_tokens) == 0:
        raise ValueError("Decoded DTW requires non-empty token sequences.")
    predicted = decode_embedding_sequence(
        decoder, predicted_tokens, pool_size=pool_size, chunk_size=chunk_size
    )
    target = decode_embedding_sequence(
        decoder, target_tokens, pool_size=pool_size, chunk_size=chunk_size
    )
    distances = pairwise_rmse(predicted, target).cpu().numpy()
    return exact_dtw(distances) / max(len(predicted_tokens), len(target_tokens))


def token_distribution_stats(tokens, codebook_size):
    """Return active codes and perplexity for valid FSQ token IDs."""
    tokens = tokens.flatten()
    tokens = tokens[(tokens >= 0) & (tokens < codebook_size)]
    if not len(tokens):
        return 0, 0.0
    counts = torch.bincount(tokens.cpu(), minlength=codebook_size).double()
    probabilities = counts[counts > 0] / counts.sum()
    perplexity = torch.exp(-(probabilities * probabilities.log()).sum()).item()
    return int((counts > 0).sum()), float(perplexity)
