import torch

from src.models.mil import AttentionMILPool, WorstSliceTopK


def test_attention_mask_ignores_padding():
    embeddings = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    pool = AttentionMILPool(8, 4, dropout=0.0)
    pooled1, weights1 = pool(embeddings, mask)

    # change padded slices and ensure output is stable
    embeddings[:, 2:] = 1000.0
    pooled2, weights2 = pool(embeddings, mask)

    assert torch.allclose(pooled1, pooled2, atol=1e-5)
    assert torch.allclose(weights1.sum(dim=1), torch.ones(2), atol=1e-6)


def test_topk_handles_fewer_valid():
    embeddings = torch.randn(1, 5, 6)
    mask = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    topk = WorstSliceTopK(6, k=3)
    out = topk(embeddings, mask)
    assert out.shape == (1,)

    # ensure padding changes do not affect output
    embeddings[:, 1:] = 999.0
    out2 = topk(embeddings, mask)
    assert torch.allclose(out, out2, atol=1e-5)
