import torch

from models.vector_lstm import VectorLSTM


def test_vector_lstm_output_shape():
    batch_size = 2
    seq_len = 3
    input_dim = 128
    model = VectorLSTM(input_dim=input_dim, latent_dim=32)
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)
    assert y.shape == (batch_size, seq_len, input_dim)

