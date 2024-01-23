from .base_tests import UnitTest, string_utils, test_results_to_score, CompositeTest
import torch
from ..util.notebook_util import count_parameters
from ..util.transformer_util import create_causal_mask
import numpy as np


class AttentionHeadMultipleDimensionsOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import ScaledDotAttention

        n_dim = np.random.randint(low=1, high=3)
        sequence_length_context = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        attention = ScaledDotAttention(d_k=d_k)
        size = [np.random.randint(low=1, high=10) for _ in range(n_dim)]

        size_queries = torch.Size(size + [sequence_length, d_k])
        size_keys = torch.Size(size + [sequence_length_context, d_k])
        size_values = torch.Size(size + [sequence_length_context, d_v])

        random_queries = torch.rand(size=size_queries)
        random_keys = torch.rand(size=size_keys)
        random_values = torch.rand(size=size_values)

        outputs = attention(random_queries, random_keys, random_values)
        self.result = outputs.shape
        self.expected = torch.Size(size + [sequence_length, d_v])

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class AttentionHeadSingleOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import ScaledDotAttention

        sequence_length_context = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        attention = ScaledDotAttention(d_k=d_k)

        size_queries = torch.Size([sequence_length, d_k])
        size_keys = torch.Size([sequence_length_context, d_k])
        size_values = torch.Size([sequence_length_context, d_v])

        random_queries = torch.rand(size=size_queries)
        random_keys = torch.rand(size=size_keys)
        random_values = torch.rand(size=size_values)

        outputs = attention(random_queries, random_keys, random_values)
        self.result = outputs.shape
        self.expected = torch.Size([sequence_length, d_v])

    def test(self):
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class AttentionHeadSoftmaxTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import ScaledDotAttention, SCORE_SAVER

        sequence_length_context = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        attention = ScaledDotAttention(d_k=d_k)

        size_queries = torch.Size([sequence_length, d_k])
        size_keys = torch.Size([sequence_length_context, d_k])
        size_values = torch.Size([sequence_length_context, d_v])

        random_queries = torch.rand(size=size_queries)
        random_keys = torch.rand(size=size_keys)
        random_values = torch.rand(size=size_values)

        SCORE_SAVER.record_scores()
        attention(random_queries, random_keys, random_values)
        scores = SCORE_SAVER.get_scores()[-1]

        self.result = torch.sum(scores, dim=-1)
        self.expected = torch.ones(size=(sequence_length,))

    def test(self):
        return torch.allclose(self.result, self.expected, atol=1e-2)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Scores do not sum up to one! Have a closer look at the Softmax Dimension.".split())


class AttentionHeadBatchedSoftmaxTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import ScaledDotAttention, SCORE_SAVER

        n_dim = np.random.randint(low=1, high=3)
        sequence_length_context = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        attention = ScaledDotAttention(d_k=d_k)
        size = [np.random.randint(low=1, high=10) for _ in range(n_dim)]

        size_queries = torch.Size(size + [sequence_length, d_k])
        size_keys = torch.Size(size + [sequence_length_context, d_k])
        size_values = torch.Size(size + [sequence_length_context, d_v])

        random_queries = torch.rand(size=size_queries)
        random_keys = torch.rand(size=size_keys)
        random_values = torch.rand(size=size_values)

        SCORE_SAVER.record_scores()
        attention(random_queries, random_keys, random_values)

        scores = SCORE_SAVER.get_scores()[-1]

        self.result = torch.sum(scores, dim=-1)
        self.expected = torch.ones(size=torch.Size(size + [sequence_length]))

    def test(self):
        return torch.allclose(self.result, self.expected, atol=1e-2)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Scores do not sum up to one! Have a closer look at the Softmax Dimension.".split())


class MultiHeadAttentionOutputShapeTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import MultiHeadAttention

        batch_size = np.random.randint(low=30, high=100)
        sequence_length_context = np.random.randint(low=30, high=100)
        sequence_length = np.random.randint(low=30, high=100)
        d_model = np.random.randint(low=30, high=100)
        d_k = np.random.randint(low=30, high=100)
        d_v = np.random.randint(low=30, high=100)
        n_heads = np.random.randint(low=1, high=10)
        multi_head_attention = MultiHeadAttention(d_model=d_model,
                                                       d_k=d_k,
                                                       d_v=d_v,
                                                       n_heads=n_heads)

        random_input = torch.rand(size=(batch_size, sequence_length, d_model))
        random_context = torch.rand(size=(batch_size, sequence_length_context, d_model))

        outputs = multi_head_attention(random_input, random_context, random_context)

        self.result = outputs.shape
        self.expected = torch.Size([batch_size, sequence_length, d_model])

    def test(self):
        
        return self.result == self.expected

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected shape {self.expected}, got shape {self.result}.".split())


class MultiHeadAttentionParameterCountTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import MultiHeadAttention

        self.d_model = np.random.randint(low=30, high=100)
        self.d_k = np.random.randint(low=30, high=100)
        self.d_v = np.random.randint(low=30, high=100)
        self.n_heads = np.random.randint(low=1, high=10)
        self.multi_head_attention = MultiHeadAttention(d_model=self.d_model,
                                                       d_k=self.d_k,
                                                       d_v=self.d_v,
                                                       n_heads=self.n_heads)

    def compute_parameters(self):
        return self.n_heads * (
                2 * self.d_model * self.d_k + self.d_model * self.d_v) + self.d_model * self.n_heads * self.d_v

    def test(self):
        expected = self.compute_parameters()
        result = count_parameters(self.multi_head_attention)
        return result == expected

    def define_failure_message(self):
        expected = self.compute_parameters()
        result = count_parameters(self.multi_head_attention)
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Expected {expected} learnable parameters, got {result}.\
             Please check your model architecture! (Are you using biases?)".split())


class AttentionPaddingTest(UnitTest):
    def __init__(self):
        super().__init__()

        from ..network.Transformer import ScaledDotAttention, SCORE_SAVER

        sequence_length = np.random.randint(low=30, high=50)
        d_k = np.random.randint(low=30, high=100)
        attention = ScaledDotAttention(d_k=d_k)
        random_input = torch.rand(size=(sequence_length, d_k))
        mask = create_causal_mask(sequence_length).squeeze(0)
        
        SCORE_SAVER.record_scores()
        attention(random_input, random_input, random_input, mask)
        scores = SCORE_SAVER.get_scores()[-1]
        self.result = scores * ~mask
        self.expected = torch.zeros_like(scores)

    def test(self):
        return torch.allclose(self.result, self.expected, atol=1e-2)

    def define_failure_message(self):
        return " ".join(f"{self.test_name} {self.failed_msg} {string_utils.ARROW}\
            Masked Softmax not implemented correctly.".split())


class TestTask2(CompositeTest):
    def define_tests(self):
        return [AttentionHeadMultipleDimensionsOutputShapeTest(),
                AttentionHeadSingleOutputShapeTest(),
                AttentionHeadSoftmaxTest(),
                AttentionHeadBatchedSoftmaxTest()]


class TestTask3(CompositeTest):
    def define_tests(self):
        return [MultiHeadAttentionParameterCountTest(),
                MultiHeadAttentionOutputShapeTest()]


class TestTask8(CompositeTest):
    def define_tests(self):
        return [AttentionPaddingTest()]


def test_task_2():
    test = TestTask2()
    return test_results_to_score(test())


def test_task_3():
    test = TestTask3()
    return test_results_to_score(test())


def test_task_8():
    test = TestTask8()
    return test_results_to_score(test())
