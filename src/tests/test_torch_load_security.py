import os
import pickle
import pytest
import torch

from app.visual_bge.eva_clip.factory import load_state_dict


class UnsafeExploit:
    def __reduce__(self):
        return (os.system, ("echo VULNERABLE",))


def test_load_state_dict_safe_weights(tmp_path):
    checkpoint_file = tmp_path / "safe_checkpoint.pt"
    safe_data = {"model": {"weight": torch.tensor([1.0, 2.0])}}
    torch.save(safe_data, checkpoint_file)

    loaded = load_state_dict(str(checkpoint_file), map_location="cpu")
    assert "weight" in loaded
    assert torch.equal(loaded["weight"], torch.tensor([1.0, 2.0]))


def test_load_state_dict_rejects_unsafe_pickle(tmp_path):
    checkpoint_file = tmp_path / "unsafe_checkpoint.pt"
    unsafe_data = {"model": UnsafeExploit()}

    # Save using standard pickle to simulate a malicious checkpoint
    with open(checkpoint_file, "wb") as f:
        pickle.dump(unsafe_data, f)

    with pytest.raises(Exception):
        load_state_dict(str(checkpoint_file), map_location="cpu")
