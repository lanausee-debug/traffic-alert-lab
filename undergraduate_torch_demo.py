"""Synthetic tensor lesson using existing model classes, without optimizer updates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from fresh_holdout_pilot import MaskedGraphTCN
from congestion_risk_pilot import RiskModel

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lesson():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261013)
    shapes = {}
    graph = np.eye(24, dtype=np.float32)  # Teaching graph, not the traffic graph.
    forecaster = MaskedGraphTCN(12, 3, graph)
    risk = RiskModel(51, nonlinear=True)
    forecaster.eval()
    before_fore = {k: v.detach().clone() for k, v in forecaster.state_dict().items()}
    before_risk = {k: v.detach().clone() for k, v in risk.state_dict().items()}
    handles = []
    def hook(name):
        def capture(module, inputs, output):
            shapes[name] = {'input': list(inputs[0].shape), 'output': list(output.shape)}
        return capture
    handles.append(forecaster.project.register_forward_hook(hook('channel_projection')))
    for i, conv in enumerate(forecaster.temporal):
        handles.append(conv.register_forward_hook(hook(f'temporal_dilation_{(1, 2, 4)[i]}')))
    for i, layer in enumerate(risk.layers):
        if isinstance(layer, torch.nn.Linear):
            handles.append(layer.register_forward_hook(hook(f'risk_linear_{i}')))
    z = torch.linspace(-1, 1, 2 * 12 * 24).reshape(2, 12, 24)
    mask = torch.ones_like(z)
    mask[:, :, 0] = 0
    calendar = torch.zeros(2, 4)
    with torch.no_grad():
        prediction = forecaster(z, mask, calendar)
        hidden_changed = z.clone()
        hidden_changed[mask == 0] = 999
        changed_prediction = forecaster(hidden_changed, mask, calendar)
        all_missing = forecaster(z, torch.zeros_like(mask), calendar)
    assert prediction.shape == (2, 3, 24) and torch.isfinite(prediction).all()
    assert torch.equal(prediction, changed_prediction)
    assert torch.isfinite(all_missing).all()
    # Separate synthetic risk features/labels, not features or labels from traffic data.
    x = torch.linspace(-2, 2, 48 * 51).reshape(48, 51)
    target = (torch.arange(48) % 2).float()
    risk.zero_grad(set_to_none=True)
    logits = risk(x)
    assert logits.shape == target.shape == (48,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    gradient_norm = sum(float(p.grad.square().sum()) for p in risk.parameters()) ** .5
    assert torch.isfinite(loss) and gradient_norm > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in risk.parameters())
    for model, before in ((forecaster, before_fore), (risk, before_risk)):
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    for handle in handles:
        handle.remove()
    return {
        'purpose': 'synthetic_tensor_and_gradient_lesson_not_traffic_experiment',
        'torch_version': torch.__version__, 'device': 'cpu', 'seed': 20261013,
        'input_shapes': {'history': [2, 12, 24], 'mask': [2, 12, 24], 'calendar': [2, 4],
                         'synthetic_risk_features': [48, 51], 'synthetic_risk_target': [48]},
        'module_shapes': shapes, 'forecast_shape': list(prediction.shape), 'risk_logit_shape': list(logits.shape),
        'forecaster_parameters': sum(p.numel() for p in forecaster.parameters()),
        'risk_mlp_parameters': sum(p.numel() for p in risk.parameters()),
        'masked_value_change_left_prediction_identical': True, 'all_missing_output_finite': True,
        'synthetic_bce': float(loss.detach()), 'synthetic_risk_gradient_norm': gradient_norm,
        'gradients_computed': True, 'optimizer_updates': 0, 'model_weights_changed': False,
        'traffic_data_read': False, 'saved_research_weights_loaded': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/undergraduate-defense/torch_demo.json')
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)
    # Reject accidental calls to research data/weight readers inside the lesson.
    with (patch('h5py.File', side_effect=RuntimeError('Synthetic lesson must not open HDF5')),
         patch('numpy.load', side_effect=RuntimeError('Synthetic lesson must not load data caches')),
         patch('torch.load', side_effect=RuntimeError('Synthetic lesson must not load weights'))):
        result = lesson()
    result['source_sha256'] = {p: sha(ROOT / p) for p in
                              ('undergraduate_torch_demo.py', 'fresh_holdout_pilot.py', 'congestion_risk_pilot.py')}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
