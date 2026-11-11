"""Actual public source + synthetic raw C/R5; no human efficacy claim."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from tests.support.brainsync import write_bids_session
from transfer.calibrated import load_calibrated_predictor
sys.path.insert(0, str(ROOT.parent / 'BrainSync_Tool_newest/BrainSync_Tool'))
from gtn_workstation.backend import CalibrationSettings, ModelTask

OUT = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else 'workflow')
OUT.mkdir(exist_ok=True)
cal = write_bids_session(OUT / 'cal', session_id='synthetic-basic-cal', target=3,
                         started_utc='2026-11-02T08:00:00Z', blocks=(tuple(range(1, 10)) * 5,))
held = write_bids_session(OUT / 'held', session_id='synthetic-basic-held', target=7,
                          started_utc='2026-11-02T09:00:00Z', blocks=(tuple(range(1, 10)) * 5,))
model = OUT / 'calibrate/personal_model.pt'

def run(task, *arguments):
    settings = CalibrationSettings(calibration_profile='old_basic_nn_source',
                                  source_checkpoint=str(ROOT / 'models/source/basic12_r1s0m0_u27950.pt'))
    sessions = (cal,) if task == 'calibrate' else (held,) if task == 'infer' else ()
    request = ModelTask.create(ROOT, task, OUT / task, artifact=model,
                               sessions=sessions, settings=settings)
    subprocess.run([request.program, *request.arguments], cwd=ROOT, check=True)
    return request.read_result()

fit = run('calibrate', '--calibration-profile', 'old_basic_nn_source', '--session-dir', cal)
assert fit['status'] == 'completed' and fit['artifact_reload_verified']
assert fit['calibration_profile']['id'] == 'old_basic_nn_source'
assert fit['internal_validation']['mode'] == 'fixed_unified_program'
assert fit['ocular_correction_applied'] is False and fit['epoch_qc_applied'] is False
loaded = load_calibrated_predictor(model)
recipe = loaded.metadata['recipe']
assert recipe['epochs'] == 90 and recipe['batch_size'] == 512
assert recipe['bn_buffers'] == 'calibration' and recipe['input_statistics'] == 'target_prefix'
assert loaded.metadata['input_request']['reference_channel'] == 'CZ'
assert tuple(loaded.metadata['input_contract']['channel_names']) == ('FZ', 'P3', 'PZ', 'P4', 'PO8', 'OZ')
inspect = run('inspect')
assert inspect['calibration_profile'] == fit['calibration_profile']
result = run('infer', '--session-dir', held)
assert result['protocol_compliant'] and result['target_labels_used'] is False
assert result['status'] in {'predicted', 'abstained'}
assert loaded.load_inference_session(held).y is None
record = {'status': 'passed', 'source': fit['calibration_profile'],
          'synthetic_contract_only': True, 'epochs': recipe['epochs'],
          'R5_status': result['status'], 'protocol_compliant': result['protocol_compliant'],
          'target_labels_used': result['target_labels_used']}
(OUT.parent / 'workflow_verification.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
