"""Verification logic correctness tests."""

import pytest
import torch
from specdecode.verification.verifier import SpeculativeVerifier


def test_greedy_all_accepted():
    candidates = [10, 20, 30]
    vocab_size = 50
    # Construct target logits where argmax matches candidates exactly
    target_logits = torch.zeros(4, vocab_size)
    target_logits[0, 10] = 10.0  # matches candidate 0
    target_logits[1, 20] = 10.0  # matches candidate 1
    target_logits[2, 30] = 10.0  # matches candidate 2
    target_logits[3, 40] = 10.0  # bonus token from target

    res = SpeculativeVerifier.verify_greedy(candidates, target_logits)

    assert res.num_speculative_accepted == 3
    assert res.accepted_tokens == [10, 20, 30, 40]
    assert not res.rejection_occurred
    assert res.rejection_index is None
    assert res.acceptance_mask == [True, True, True]


def test_greedy_first_rejected():
    candidates = [10, 20, 30]
    vocab_size = 50
    target_logits = torch.zeros(4, vocab_size)
    target_logits[0, 99 % vocab_size] = 10.0  # target actually wanted token 49, not 10!

    res = SpeculativeVerifier.verify_greedy(candidates, target_logits)

    assert res.num_speculative_accepted == 0
    assert res.accepted_tokens == [49]
    assert res.rejection_occurred
    assert res.rejection_index == 0
    assert res.acceptance_mask == [False]


def test_greedy_second_rejected():
    candidates = [10, 20, 30]
    vocab_size = 50
    target_logits = torch.zeros(4, vocab_size)
    target_logits[0, 10] = 10.0  # accepts candidate 0 (token 10)
    target_logits[1, 25] = 10.0  # rejects candidate 1 (candidate had 20, target wanted 25)

    res = SpeculativeVerifier.verify_greedy(candidates, target_logits)

    assert res.num_speculative_accepted == 1
    assert res.accepted_tokens == [10, 25]
    assert res.rejection_occurred
    assert res.rejection_index == 1
    assert res.acceptance_mask == [True, False]


def test_greedy_eos_handling():
    eos_token_id = 99
    candidates = [10, 99, 30]
    vocab_size = 100
    target_logits = torch.zeros(4, vocab_size)
    target_logits[0, 10] = 10.0
    target_logits[1, 99] = 10.0  # EOS accepted

    res = SpeculativeVerifier.verify_greedy(candidates, target_logits, eos_token_id=eos_token_id)

    assert res.accepted_tokens == [10, 99]
    assert not res.rejection_occurred
