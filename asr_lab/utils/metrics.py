"""Evaluation metrics for ASR."""

from typing import Sequence


def _levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int, int]:
    """Compute Levenshtein distance with edit counts.

    Returns:
        Tuple of (distance, substitutions, insertions, deletions)
    """
    m, n = len(ref), len(hyp)

    # DP table: each cell contains (distance, subs, ins, dels)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(1, m + 1):
        dp[i][0] = (i, 0, 0, i)  # i deletions
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, j, 0)  # j insertions

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                sub = dp[i - 1][j - 1]
                sub_cost = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])

                # Insertion (extra word in hypothesis)
                ins = dp[i][j - 1]
                ins_cost = (ins[0] + 1, ins[1], ins[2] + 1, ins[3])

                # Deletion (missing word in hypothesis)
                delete = dp[i - 1][j]
                del_cost = (delete[0] + 1, delete[1], delete[2], delete[3] + 1)

                # Choose minimum
                dp[i][j] = min(sub_cost, ins_cost, del_cost, key=lambda x: x[0])

    return dp[m][n]


def compute_wer(
    references: list[str],
    hypotheses: list[str],
    return_details: bool = False,
) -> float | dict[str, float]:
    """Compute Word Error Rate (WER).

    WER = (S + I + D) / N
    where S = substitutions, I = insertions, D = deletions, N = reference words

    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
        return_details: Whether to return detailed metrics

    Returns:
        WER as a float, or dict with S/I/D counts if return_details=True
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")

    total_words = 0
    total_errors = 0
    total_subs = 0
    total_ins = 0
    total_dels = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()

        distance, subs, ins, dels = _levenshtein_distance(ref_words, hyp_words)

        total_words += len(ref_words)
        total_errors += distance
        total_subs += subs
        total_ins += ins
        total_dels += dels

    wer = total_errors / max(total_words, 1)

    if return_details:
        return {
            "wer": wer,
            "substitutions": total_subs,
            "insertions": total_ins,
            "deletions": total_dels,
            "total_words": total_words,
            "total_errors": total_errors,
        }

    return wer


def compute_cer(
    references: list[str],
    hypotheses: list[str],
    return_details: bool = False,
) -> float | dict[str, float]:
    """Compute Character Error Rate (CER).

    CER = (S + I + D) / N
    where S = substitutions, I = insertions, D = deletions, N = reference chars

    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
        return_details: Whether to return detailed metrics

    Returns:
        CER as a float, or dict with S/I/D counts if return_details=True
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")

    total_chars = 0
    total_errors = 0
    total_subs = 0
    total_ins = 0
    total_dels = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.lower())
        hyp_chars = list(hyp.lower())

        distance, subs, ins, dels = _levenshtein_distance(ref_chars, hyp_chars)

        total_chars += len(ref_chars)
        total_errors += distance
        total_subs += subs
        total_ins += ins
        total_dels += dels

    cer = total_errors / max(total_chars, 1)

    if return_details:
        return {
            "cer": cer,
            "substitutions": total_subs,
            "insertions": total_ins,
            "deletions": total_dels,
            "total_chars": total_chars,
            "total_errors": total_errors,
        }

    return cer
