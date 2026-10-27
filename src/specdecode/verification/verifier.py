"""Speculative candidate verification algorithms.
Guarantees authoritative target model semantic preservation under greedy and sampling modes.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass
class VerificationResult:
    """Outcome of speculative candidate verification."""
    accepted_tokens: List[int]
    num_speculative_accepted: int
    num_candidates_evaluated: int
    rejection_occurred: bool
    rejection_index: Optional[int]
    acceptance_mask: List[bool]


class SpeculativeVerifier:
    """Verifies candidate token sequences against authoritative target model logits."""

    @staticmethod
    def verify_greedy(
        candidate_tokens: List[int],
        target_logits: torch.Tensor,
        eos_token_id: Optional[int] = None,
    ) -> VerificationResult:
        """Verify candidate tokens greedily.

        Args:
            candidate_tokens: List of K candidate token IDs [c_1, c_2, ..., c_K]
            target_logits: Tensor of shape (K + 1, vocab_size), where
                           target_logits[0] predicts the token following prefix,
                           target_logits[i] predicts the token following candidate c_i.
            eos_token_id: Optional EOS token ID. If accepted token is EOS, verification stops.

        Returns:
            VerificationResult with list of accepted tokens, counts, and metadata.
        """
        K = len(candidate_tokens)
        accepted = []
        acceptance_mask = []
        rejection_occurred = False
        rejection_index = None

        # Authoritative greedy predictions from target logits
        target_preds = torch.argmax(target_logits, dim=-1).tolist()

        for i in range(K):
            authoritative_token = target_preds[i]
            candidate_token = candidate_tokens[i]

            if candidate_token == authoritative_token:
                accepted.append(candidate_token)
                acceptance_mask.append(True)
                if eos_token_id is not None and candidate_token == eos_token_id:
                    # EOS reached in candidate
                    return VerificationResult(
                        accepted_tokens=accepted,
                        num_speculative_accepted=len(accepted),
                        num_candidates_evaluated=i + 1,
                        rejection_occurred=False,
                        rejection_index=None,
                        acceptance_mask=acceptance_mask,
                    )
            else:
                # Candidate rejected: append the authoritative token and stop
                accepted.append(authoritative_token)
                acceptance_mask.append(False)
                rejection_occurred = True
                rejection_index = i
                return VerificationResult(
                    accepted_tokens=accepted,
                    num_speculative_accepted=i,
                    num_candidates_evaluated=i + 1,
                    rejection_occurred=True,
                    rejection_index=i,
                    acceptance_mask=acceptance_mask,
                )

        # If all K candidates were accepted, the target model also gives the bonus next token!
        bonus_token = target_preds[K]
        accepted.append(bonus_token)

        return VerificationResult(
            accepted_tokens=accepted,
            num_speculative_accepted=K,
            num_candidates_evaluated=K,
            rejection_occurred=False,
            rejection_index=None,
            acceptance_mask=acceptance_mask,
        )

    @staticmethod
    def verify_sampling(
        candidate_tokens: List[int],
        draft_probs: List[torch.Tensor],
        target_probs: torch.Tensor,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> VerificationResult:
        """Distributionally exact speculative sampling verification (Leviathan et al.).

        Acceptance probability: min(1, P_target(c) / P_draft(c))
        """
        K = len(candidate_tokens)
        accepted = []
        acceptance_mask = []
        rejection_occurred = False
        rejection_index = None

        for i in range(K):
            c_i = candidate_tokens[i]
            p_target = target_probs[i, c_i].item()
            p_draft = max(draft_probs[i][c_i].item(), 1e-12)
            alpha = min(1.0, p_target / p_draft)

            rand_val = torch.rand(1).item()
            if rand_val < alpha:
                accepted.append(c_i)
                acceptance_mask.append(True)
                if eos_token_id is not None and c_i == eos_token_id:
                    return VerificationResult(
                        accepted_tokens=accepted,
                        num_speculative_accepted=len(accepted),
                        num_candidates_evaluated=i + 1,
                        rejection_occurred=False,
                        rejection_index=None,
                        acceptance_mask=acceptance_mask,
                    )
            else:
                # Rejection sampling: sample from normalized positive difference max(0, P_target - P_draft)
                rejection_occurred = True
                rejection_index = i
                acceptance_mask.append(False)
                diff = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                diff_sum = diff.sum()
                if diff_sum > 0:
                    diff_prob = diff / diff_sum
                    authoritative_sample = torch.multinomial(diff_prob, 1).item()
                else:
                    authoritative_sample = torch.multinomial(target_probs[i], 1).item()

                accepted.append(authoritative_sample)
                return VerificationResult(
                    accepted_tokens=accepted,
                    num_speculative_accepted=i,
                    num_candidates_evaluated=i + 1,
                    rejection_occurred=True,
                    rejection_index=i,
                    acceptance_mask=acceptance_mask,
                )

        # All accepted, sample bonus token from target_probs[K]
        bonus_token = torch.multinomial(target_probs[K], 1).item()
        accepted.append(bonus_token)
        return VerificationResult(
            accepted_tokens=accepted,
            num_speculative_accepted=K,
            num_candidates_evaluated=K,
            rejection_occurred=False,
            rejection_index=None,
            acceptance_mask=acceptance_mask,
        )
