import pytest
import torch

from bgflow.nn.flow.crd_transform.pca import WhitenFlow


def test_whiten_flow_rejects_rank_deficient_full_dimensional_fit():
    base = torch.randn(32, 6)
    base[:, 3:] = base[:, :3]

    with pytest.raises(ValueError, match="nonpositive eigenvalues"):
        WhitenFlow(base, keepdims=base.shape[1], whiten_inverse=False)
