"""Generate a recording quality report without using targets or fitting any model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from data.brainsync import load_brainsync_inference_session  # noqa: E402
from data.continuous_qc import continuous_qc_from_record, load_continuous_qc  # noqa: E402
from data.epochs import PreprocessingSpec  # noqa: E402
from data.ocular import eog_regression_from_record, load_eog_regression  # noqa: E402
from experiments.brainsync_model_task import (  # noqa: E402
    _apply_reference,
    _resolve_reference_request,
    calibration_preprocessing,
)
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
        channels = None
        predictor = None
        if args.model_artifact:
            predictor = load_calibrated_predictor(args.model_artifact, device="cpu")
            preprocessing = PreprocessingSpec(**predictor.metadata["input_contract"]["preprocessing"])
            frozen = predictor.metadata["input_request"]
            continuous = continuous_qc_from_record(frozen["continuous_qc_model"]) if frozen.get("continuous_qc_model") else None
            ocular = eog_regression_from_record(frozen["ocular_model"]) if frozen.get("ocular_model") else None
            reference = _resolve_reference_request(frozen.get("reference_channel"), ocular, continuous,
                                                   mode=frozen.get("reference_mode", "auto"))
            channels = tuple(frozen["channel_names"])
            state["epoch"] = predictor.epoch_quality
            request["model_sha256"] = file_sha256(args.model_artifact)
        else:
            preprocessing = calibration_preprocessing(args.preprocessing_config)
            continuous = load_continuous_qc(args.continuous_qc_model) if args.continuous_qc_model else None
            ocular = load_eog_regression(args.ocular_model) if args.ocular_model else None
            reference = _resolve_reference_request(args.reference_channel, ocular, continuous, mode=args.reference_mode)
            state["epoch"] = load_epoch_quality(args.epoch_qc_model) if args.epoch_qc_model else None
        dataset = _apply_reference(load_brainsync_inference_session(root, channels=channels,
                                   preprocessing=preprocessing, ocular_model=ocular,
                                   continuous_qc_model=continuous), reference, ocular, channels=channels)
        if predictor is not None:
            predictor.validate_input(dataset)
            predictor.validate_inference_scope(dataset)
        return dataset

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
