from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import torch
import numpy as np
from ..util.notebook_util import count_parameters
from ..util.transformer_util import create_causal_mask


class DecoderBlockOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import DecoderBlock


        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=30, high=100)
        d_ff = np.random.randint(low=30, high=100)
        random_input = torch.rand((batch_size, sequence_length, d_model))
        random_context = torch.rand((batch_size, sequence_length, d_model))
        causal_mask = create_causal_mask(sequence_length)

        decoder_block = DecoderBlock(d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_v,
                                     n_heads=n_heads,
                                     d_ff=d_ff)

        output = decoder_block(random_input, random_context, causal_mask)
        self.result = output.shape
        self.expected = torch.Size([batch_size, sequence_length, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class DecoderBlockOutputNorm(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import DecoderBlock


        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        random_input = torch.rand((batch_size, sequence_length, d_model))
        random_context = torch.rand((batch_size, sequence_length, d_model))
        causal_mask = create_causal_mask(sequence_length)

        decoder_block = DecoderBlock(d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_v,
                                     n_heads=n_heads,
                                     d_ff=d_ff)

        output = decoder_block(random_input, random_context, causal_mask)
        mean = torch.mean(output).item()
        std = torch.std(output).item()
        self.result = np.array([mean, std])
        self.expected = np.array([0, 1])

    def test(self):
        return np.isclose(self.result, self.expected).all()

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected [Mean, Std]: {self.expected}, got: {self.result}. Please check the layer normalization!".split())


class DecoderBlockParameterCountTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import DecoderBlock

        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)

        decoder_block = DecoderBlock(d_model=d_model,
                                     d_k=d_k,
                                     d_v=d_v,
                                     n_heads=n_heads,
                                     d_ff=d_ff)

        count_ln = 2 * d_model
        count_mh = n_heads * (2 * d_model * d_k + d_model * d_v) + d_model * n_heads * d_v
        count_ffn = d_model * d_ff * 2 + d_model + d_ff

        self.expected = 3 * count_ln + 2 * count_mh + count_ffn
        self.result = count_parameters(decoder_block)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())


class DecoderOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import Decoder


        batch_size = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)
        random_input = torch.rand((batch_size, sequence_length, d_model))
        random_context = torch.rand((batch_size, sequence_length, d_model))

        encoder_stack = Decoder(d_model=d_model,
                                d_k=d_k,
                                d_v=d_v,
                                n_heads=n_heads,
                                d_ff=d_ff,
                                n=n)

        output = encoder_stack(random_input, random_context)
        self.result = output.shape
        self.expected = torch.Size([batch_size, sequence_length, d_model])

    def test(self):
        return self.expected == self.result

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class DecoderParameterCountTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import Decoder

        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        d_ff = np.random.randint(low=30, high=1000)
        n = np.random.randint(low=1, high=10)

        decoder_stack = Decoder(d_model=d_model,
                                d_k=d_k,
                                d_v=d_v,
                                n_heads=n_heads,
                                d_ff=d_ff,
                                n=n)

        count_ln = 2 * d_model
        count_mh = n_heads * (2 * d_model * d_k + d_model * d_v) + d_model * n_heads * d_v
        count_ffn = d_model * d_ff * 2 + d_model + d_ff

        self.expected = n * (3 * count_ln + 2 * count_mh + count_ffn)
        self.result = count_parameters(decoder_stack)

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {self.expected} learnable parameters, got {self.result}. Please check your model architecture!".split())


class TestTask9(CompositeTest):
    def define_tests(self, ):
        return [
            DecoderBlockOutputShapeTest(),
            DecoderBlockOutputNorm(),
            DecoderBlockParameterCountTest()
        ]


class TestTask10(CompositeTest):
    def define_tests(self, ):
        return [
            DecoderOutputShapeTest(),
            DecoderParameterCountTest()
        ]


def test_task_9():
    test = TestTask9()
    return test_results_to_score(test())


def test_task_10():
    test = TestTask10()
    return test_results_to_score(test())
