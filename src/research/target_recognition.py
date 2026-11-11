"""R1/R5 nine-choice recognition with scored accuracy and explicit coverage."""
from __future__ import annotations

from collections import Counter

import numpy as np

from data.candidate_task import MembershipDecision
from models.candidate_evidence import CandidateEvidence, membership_decision_scores
from research.candidate_metrics import scored_target_accuracy
from research.contracts import semantic_sha256
from transfer.ensemble import EnsembleProgram, EnsembleScorer

VOCABULARY = tuple(map(str, range(1, 10)))
LEAVES = ("nn_source", "waveform_lda", "xdawn_erp")
DISPLAYED_METHODS = ("nn_source", "nn_scratch", "xdawn_erp", "three_experts")
REQUIRED_EXPERTS = frozenset((*LEAVES, "nn_scratch"))


def summarize_target_decisions(records):
    records = tuple(records)
    if len({r["decision_id"] for r in records}) != len(records):
        raise ValueError("recognition records must contain each requested decision exactly once")
    metric = scored_target_accuracy([r["prediction"] for r in records], [r["target"] for r in records],
        candidate_vocabulary=VOCABULARY).record()
    metric["unscored_reasons"] = dict(Counter(r["reason"] or r["status"] for r in records if r["prediction"] is None))
    metric["random_baseline"] = 1 / 9
    return metric


def score_nine_choice_recording(*, session_id, participant_key, event_ids, candidate_ids,
                                event_logits, expert_identities, raw_support_digest, revealed_target):
    """Score independent nine-flash cycles and five-cycle blocks before truth evaluation.

    Each expert supplies one score or NaN per original scheduled event. Missing
    evidence never changes the requested physical schedule or becomes a zero logit.
    """
    ids = np.asarray(event_ids).astype(str)
    codes = np.asarray(candidate_ids).astype(str)
    if (ids.ndim != 1 or not len(ids) or len(ids) % 45 or codes.shape != ids.shape
            or len(set(ids)) != len(ids) or set(codes) != set(VOCABULARY)):
        raise ValueError("nine-choice scoring requires complete original 45-flash block ledgers")
    if any(set(codes[i:i+9]) != set(VOCABULARY) for i in range(0, len(ids), 9)):
        raise ValueError("each physical cycle must contain each of the nine candidates exactly once")
    if set(event_logits) != REQUIRED_EXPERTS or set(expert_identities) != REQUIRED_EXPERTS:
        raise ValueError("the fixed four-expert fit roster includes internal waveform LDA")
    values = {name: np.asarray(vector, dtype=float) for name, vector in event_logits.items()}
    if any(vector.shape != ids.shape for vector in values.values()):
        raise ValueError("expert scores must align with every original scheduled event")
    program = EnsembleProgram.equal_weights("source_LDA_xDAWN", LEAVES)
    pool = EnsembleScorer(program)
    panels = {}
    for repetitions in (1, 5):
        by_method = {method: [] for method in DISPLAYED_METHODS}
        width = repetitions * 9
        for start in range(0, len(ids), width):
            stop = start + width
            block, cycle = start // 45 + 1, start % 45 // 9 + 1
            decision_id = f"{session_id}:R{repetitions}:block:{block}" + (f":cycle:{cycle}" if repetitions == 1 else "")
            decision = MembershipDecision.direct(codes[start:stop], VOCABULARY,
                event_ids=ids[start:stop], repetition_indices=np.repeat(np.arange(repetitions), 9),
                repetitions=repetitions, evidence_indices=np.arange(width))
            support = semantic_sha256({"raw_support_digest": raw_support_digest,
                "event_ids": ids[start:stop].tolist(), "candidate_ids": codes[start:stop].tolist()})
            evidence = {}
            for name in REQUIRED_EXPERTS:
                scores = membership_decision_scores(values[name][start:stop], decision,
                    available=np.isfinite(values[name][start:stop]))
                evidence[name] = CandidateEvidence.from_membership_scores(scores, decision,
                    participant_key=participant_key, decision_id=decision_id, physical_support_digest=support,
                    expert_identities=(expert_identities[name],))
            evidence["three_experts"] = pool.score({name: evidence[name] for name in LEAVES})
            # The revealed target enters only after every expert and pool chose.
            for name in DISPLAYED_METHODS:
                item = evidence[name]
                by_method[name].append({"decision_id": decision_id, "session_id": session_id,
                    "block": block, "cycle": cycle if repetitions == 1 else None,
                    "evidence_level": repetitions, "target": str(revealed_target), "prediction": item.predicted,
                    "status": item.status, "reason": item.failure_reason,
                    "probability_available": item.probability_available, "evidence": item.record()})
        panels[f"R{repetitions}"] = {name: {"metrics": summarize_target_decisions(records), "decisions": records}
                                     for name, records in by_method.items()}
    return {"schema": "n2p3_scored_nine_choice_recording/1", "pool_program": program.record(), "panels": panels}


def compare_common_scored_decisions(left, right):
    """Describe the same actually scored requests alongside each arm's full coverage."""
    a, b = ({r["decision_id"]: r for r in records} for records in (left, right))
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("paired recognition requires the same unique requested decisions")
    if any(a[key]["target"] != b[key]["target"] for key in a):
        raise ValueError("paired revealed targets disagree")
    common = [key for key in a if a[key]["prediction"] is not None and b[key]["prediction"] is not None]
    return {"requested_decisions": len(a), "common_scored_decisions": len(common),
        "left": summarize_target_decisions([a[key] for key in common]),
        "right": summarize_target_decisions([b[key] for key in common]), "common_decision_ids": common}
