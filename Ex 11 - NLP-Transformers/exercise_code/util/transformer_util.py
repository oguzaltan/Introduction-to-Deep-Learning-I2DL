import torch


class AttentionScoresSaver:
    """
    Module to save and visualize scores
    """

    def __init__(self):
        self._save_scores = False
        self._scores = []

    def save(self, score):
        if self._save_scores:
            self._scores.append(score)

    def record_scores(self):
        self._save_scores = True

    def reset(self):
        self._scores = []
        self._save_scores = False

    def get_scores(self, reset=True):
        scores = self._scores
        self.reset()
        return scores


def positional_encoding(d_model: int,
                        max_length: int) -> torch.Tensor:
    """
    Computes the positional encoding matrix
    Args:
        d_model: Dimension of Embedding
        max_length: Maximums sequence length

    Shape:
        - output: (max_length, d_model)
    """
    output = None

    i = torch.arange(0, d_model, 2) / d_model
    pos = torch.arange(0, max_length)[:, None]

    angle_freq = torch.exp(i * (-torch.log(torch.Tensor([10000]))))

    output = torch.zeros((max_length, d_model))

    output[:, 0::2] = torch.sin(pos * angle_freq)
    output[:, 1::2] = torch.cos(pos * angle_freq)

    return output


def create_causal_mask(decoder_length: int) -> torch.Tensor:
    """
    Creates a lower triangle boolean mask for decoder self attention.
    Args:
        decoder_length: Sequence length of decoder

    Shape:
        - output: (batch_size, sequence_length, sequence_length)
    """
    output = torch.ones((decoder_length, decoder_length))
    output = torch.tril(output, diagonal=0).bool()

    return output.unsqueeze_(0)
