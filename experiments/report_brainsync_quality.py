"""Generate a recording quality report without using targets or fitting any model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from data.brainsync import (  # noqa: E402
    BRAIN_SYNC_PREPROCESSING,
    apply_brainsync_reference,
    load_brainsync_inference_session,
    resolve_brainsync_reference,
)
from data.continuous_qc import load_continuous_qc  # noqa: E402
from data.epochs import load_preprocessing_spec  # noqa: E402
from data.ocular import load_eog_regression  # noqa: E402
from research.recording_quality import create_recording_quality_report, file_sha256  # noqa: E402
from transfer.calibrated import load_calibrated_predictor  # noqa: E402
from transfer.quality import load_epoch_quality  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path)
    parser.add_argument("--continuous-qc-model", type=Path)
    parser.add_argument("--epoch-qc-model", type=Path)
    parser.add_argument("--ocular-model", type=Path)
    parser.add_argument("--preprocessing-config", type=Path,
                        help="Complete explicit preprocessing JSON; otherwise use current BrainSync defaults")
    parser.add_argument("--reference-channel")
    parser.add_argument("--reference-mode", choices=("auto", "native"), default="auto")
    args = parser.parse_args(argv)
    if args.model_artifact and (args.continuous_qc_model or args.epoch_qc_model or args.ocular_model or args.preprocessing_config or args.reference_channel or args.reference_mode != "auto"):
        parser.error("a frozen personal model cannot be combined with preprocessing overrides")
    state = {}
    request = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
               if name not in {"session_dir", "output_dir"}}

    def loader(root):
        if args.model_artifact:
            predictor = load_calibrated_predictor(args.model_artifact, device="cpu")
            state["epoch"] = predictor.epoch_quality
            request["model_sha256"] = file_sha256(args.model_artifact)
            return predictor.load_inference_session(root)
        preprocessing = load_preprocessing_spec(args.preprocessing_config) if args.preprocessing_config else BRAIN_SYNC_PREPROCESSING
        continuous = load_continuous_qc(args.continuous_qc_model) if args.continuous_qc_model else None
        ocular = load_eog_regression(args.ocular_model) if args.ocular_model else None
        reference = resolve_brainsync_reference(args.reference_channel, ocular, continuous, mode=args.reference_mode)
        state["epoch"] = load_epoch_quality(args.epoch_qc_model) if args.epoch_qc_model else None
        return apply_brainsync_reference(
            load_brainsync_inference_session(root, preprocessing=preprocessing, ocular_model=ocular,
                                            continuous_qc_model=continuous), reference, ocular,
        )

    def progress(value, message):
        print(f"PROGRESS {value}", flush=True)
        print(message, flush=True)

    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        return create_recording_quality_report(args.session_dir, args.output_dir, loader=loader,
                                               epoch_quality=lambda: state.get("epoch"), progress=progress, request=request)


if __name__ == "__main__":
    main()
