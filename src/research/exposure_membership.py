"""One physical exposure's target indicator on an explicitly ordered vocabulary.

Known sets and blanks carry observed membership. Only a missing row consumes
the caller's declared latent single-identity assignment. This grants no data
authority and adds no discrete state or probability mass.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np

from research.frozen_metadata import freeze_metadata
from transfer.noise_model import metadata_digest

IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _candidate_identity(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, Integral)) or value == "":
        raise ValueError("physical membership candidates use distinct nonempty string or integer identities")
    return int(value) if isinstance(value, Integral) else value


class MissingExposureIdentityError(ValueError):
    def __init__(self, exposure):
        self.exposure_id = exposure["event_id"]
        self.candidate_options = tuple(exposure["candidate_options"])
        super().__init__("actual exposure needs its explicit latent identity/law; no implicit stimulus prior")


class ExposureCandidateMembership:
    """Freeze the effective event-by-candidate law before design or cache use.

    An explicit axis comes from the original history producer. Legacy one-hot
    inputs instead establish their observed columns by candidate_id. Known
    physical IDs/options are optional provenance once the full axis is given.
    """

    def __init__(self, exposures, candidate_vocabulary, *, declared_vocabulary=None):
        vocabulary = tuple(_candidate_identity(q) for q in candidate_vocabulary)
        if len(vocabulary) < 2 or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("physical membership requires a complete ordered vocabulary of unique candidate identities")
        positions = {q: column for column, q in enumerate(vocabulary)}
        if declared_vocabulary is not None and tuple(_candidate_identity(q) for q in declared_vocabulary) != vocabulary:
            raise ValueError("physical membership columns differ from their original declared ordered candidate vocabulary")
        rows, options, order = {}, {}, []
        for event in exposures:
            identifier = event["event_id"]
            if identifier in rows:
                raise ValueError("each actual physical exposure needs its own original event identity")
            order.append(identifier)
            if event["membership"] is None:
                possible = tuple(_candidate_identity(q) for q in event["candidate_options"])
                if (not possible or len(set(possible)) != len(possible)
                        or not set(possible) <= set(vocabulary)):
                    raise ValueError("unknown single-identity exposure needs nonempty unique original candidate options")
                rows[identifier], options[identifier] = None, possible
                continue
            row = np.asarray(event["membership"])
            if (row.ndim != 1 or row.shape != (len(vocabulary),) or row.dtype.kind not in "biuf"
                    or not np.isfinite(row).all() or not ((row == 0) | (row == 1)).all()):
                raise ValueError("known physical membership must be one actual binary row on the complete ordered vocabulary")
            row = tuple(bool(bit) for bit in row)
            if declared_vocabulary is None:
                legacy = event.get("candidate_id")
                if (not isinstance(legacy, (str, Integral)) or isinstance(legacy, (bool, np.bool_))
                        or legacy not in positions or row != tuple(q == legacy for q in vocabulary)):
                    raise ValueError("known set or blank requires its original ordered membership vocabulary; legacy one-hot IDs must match their columns")
            rows[identifier] = row
        self.vocabulary = vocabulary
        self._positions = MappingProxyType(positions)
        self._rows, self._options = freeze_metadata(rows), freeze_metadata(options)
        self.record = freeze_metadata({"schema": "physical_event_candidate_membership/1",
            "candidate_vocabulary": vocabulary, "exposure_order": order,
            "observed_membership_by_event": rows, "unknown_single_identity_options": options,
            "column_identity_origin": ("original_history_ordered_vocabulary" if declared_vocabulary is not None
                else "legacy_onehot_candidate_id_checked_against_every_known_row"),
            "known_physical_ID_and_options_role": "optional_provenance_not_target_membership",
            "common_response_count_per_actual_exposure": 1,
            "new_states_or_prior_mass_added": False, "raw_or_C_authority_granted": False,
            "producer_sha256": IMPLEMENTATION_SHA256})
        self.identity = metadata_digest(self.record)

    def target_indicator(self, exposure_id, candidate, unknown_assignments=None):
        candidate = _candidate_identity(candidate)
        if candidate not in self._positions:
            raise ValueError("physical target indicator requires an original complete-vocabulary candidate")
        row = self._rows[exposure_id]
        if row is not None:
            return row[self._positions[candidate]], "original_public_membership"
        assignments = {} if unknown_assignments is None else unknown_assignments
        assigned = assignments.get(exposure_id)
        if assigned is not None:
            assigned = _candidate_identity(assigned)
        possible = self._options[exposure_id]
        if assigned not in possible:
            raise MissingExposureIdentityError({"event_id": exposure_id, "candidate_options": possible})
        return assigned == candidate, "given_latent_exposure_state_no_prior_applied"
