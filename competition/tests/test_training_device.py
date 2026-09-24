import pytest
import torch

from train_specialist import resolve_device


def test_training_device_is_explicit_and_does_not_fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device('cpu').type == 'cpu'
    assert resolve_device('auto').type == 'cpu'
    with pytest.raises(ValueError, match='CUDA was requested'):
        resolve_device('cuda')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert resolve_device('auto').type == 'cuda'
    assert resolve_device('cuda').type == 'cuda'
    with pytest.raises(ValueError, match='Device must be'):
        resolve_device('gpu')
