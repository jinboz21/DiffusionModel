"""Compare the recovered CNN and cached GPROF-NN on identical Korea pixels.

Run from the project root. This performs inference only; it never trains or downloads.
CNN predictions are cached per scene so an interrupted run can be resumed.
"""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import hdf5plugin  # noqa: F401 -- required for some SatRain NetCDF compression
import numpy as np
import pandas as pd
import torch
import xarray as xr
from satrain.evaluation import Evaluator
from satrain.input import GMI
from satrain.pytorch import PytorchRetrieval
from satrain.target import TargetConfig

from cnn_official.official_model import IPWGUNet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / (
    'cnn_official/runs/colab_gmi_20261105_221329/checkpoints/best-epoch=18.ckpt'
)
DIMS = ('latitude', 'longitude')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def grid_values(dataset, name, target):
    value = dataset[name].transpose(*DIMS)
    if value.shape != target.surface_precip.transpose(*DIMS).shape:
        raise ValueError(f'{name}: grid shape differs from the reference.')
    for dim in DIMS:
        if dim in value.coords and dim in target.coords:
            np.testing.assert_allclose(value[dim], target[dim], rtol=0, atol=1e-6)
    return value.values.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'experiments/gprof_korea/manifest.json')
    parser.add_argument('--gprof-evaluation', type=Path,
                        default=ROOT / 'experiments/gprof_korea/full/evaluation')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/cnn_gprof_korea_comparison')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive.')

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    summary_path = args.gprof_evaluation / 'summary.json'
    gprof_summary = json.loads(summary_path.read_text(encoding='utf-8'))
    scenes = manifest['scenes']
    if (manifest['domain'] != 'korea' or manifest['split'] != 'testing'
            or manifest['base_sensor'] != 'gmi' or len(scenes) != 600
            or manifest['official_scene_count'] != 600
            or not gprof_summary['complete_test'] or gprof_summary['scenes'] != 600):
        raise ValueError('This comparison requires both complete Korea 600-scene datasets.')
    if len({scene['timestamp'] for scene in scenes}) != 600:
        raise ValueError('Duplicate scene IDs in manifest.')

    # Check all required paths before starting the potentially lengthy CPU inference.
    file_records = []
    for scene in scenes:
        stamp = scene['timestamp']
        target = Path(scene['target_gridded'])
        scene['gmi_gridded'] = str(target.with_name(f'gmi_{stamp}.nc'))
        scene['paired_gprof'] = str(args.gprof_evaluation / 'gridded' / f'paired_{stamp}.nc')
        for key in ('target_gridded', 'target_on_swath', 'gmi_gridded', 'paired_gprof'):
            path = Path(scene[key])
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            file_records.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    target_config = TargetConfig(**gprof_summary['target_config'])
    provenance = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': sha256(args.checkpoint),
        'comparison_code_sha256': sha256(__file__),
        'model_code_sha256': sha256(Path(__file__).with_name('official_model.py')),
        'manifest_sha256': sha256(args.manifest),
        'gprof_summary_sha256': sha256(summary_path),
        'gprof_experiment_signature': gprof_summary['experiment_signature'],
        'files_metadata_sha256': hashlib.sha256(json.dumps(file_records).encode()).hexdigest(),
        'versions': {name: version(name) for name in ('satrain', 'torch', 'numpy', 'lightning')},
        'target_config': vars(target_config),
        'input': 'GMI 13 channels; minmax; nan=-1.5; include_angles=False',
        'geometry': 'gridded', 'tile_size': 256, 'overlap': 64,
        'batch_size': args.batch_size, 'threads': args.threads, 'device': 'cpu', 'dtype': 'float32',
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    output = args.output / signature[:16]
    cache = output / 'cnn_predictions'
    cache.mkdir(parents=True, exist_ok=True)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('Output:', output, flush=True)
    print('Inference only: 600 scenes, CPU float32. Existing scene caches will be reused.', flush=True)

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = IPWGUNet(input_features=13, internal_features=[32, 64, 128, 256, 512], n_epochs=20)
    # This default is the user's own recovered Colab checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.eval()
    inputs = [GMI(normalize='minmax', nan=-1.5, include_angles=False)]
    retrieval = PytorchRetrieval(model, retrieval_input=inputs, stack=True,
                                device=torch.device('cpu'), dtype=torch.float32)
    evaluator = Evaluator(base_sensor='gmi', domain='korea', geometry='gridded',
                          retrieval_input=inputs, target_config=target_config,
                          data_path=Path(manifest['data_path']), download=False)
    # Pair explicitly by timestamp, independently of filesystem discovery order.
    for source in ('gmi', 'atms', 'ancillary', 'geo', 'geo_t', 'geo_ir', 'geo_ir_t'):
        for geometry in ('gridded', 'on_swath'):
            attribute = f'{source}_{geometry}'
            if hasattr(evaluator, attribute):
                delattr(evaluator, attribute)
    evaluator.target_gridded = [Path(s['target_gridded']) for s in scenes]
    evaluator.target_on_swath = [Path(s['target_on_swath']) for s in scenes]
    evaluator.gmi_gridded = [Path(s['gmi_gridded']) for s in scenes]

    names = ('CNN_raw', 'CNN_clipped', 'GPROF-NN V8 NONE', 'Always_zero')
    totals = {name: np.zeros(3, dtype=np.float64) for name in names}  # abs, squared, signed error
    counts = {(name, threshold): np.zeros(3, dtype=np.int64)
              for name in names for threshold in (0.1, 10.0)}
    coverage = []
    n_pixels, reference_sum, negative_count = 0, 0.0, 0

    for index, scene in enumerate(scenes):
        stamp = scene['timestamp']
        with xr.open_dataset(scene['target_gridded'], engine='h5netcdf') as source:
            target = source.load()
        reference = target.surface_precip.transpose(*DIMS).values.copy()
        eligible = (target.pixel_index.transpose(*DIMS).values >= 0) & ~target_config.get_mask(target)
        with xr.open_dataset(scene['paired_gprof']) as paired:
            if paired.attrs.get('experiment_signature') != gprof_summary['experiment_signature']:
                raise ValueError(f'GPROF provenance mismatch: {stamp}')
            gprof = grid_values(paired, 'surface_precip', target)
            gprof_mask = grid_values(paired, 'common_valid_mask', target).astype(bool)
            np.testing.assert_array_equal(gprof_mask, eligible & np.isfinite(gprof))
            paired_reference = grid_values(paired, 'surface_precip_ref', target)
            np.testing.assert_allclose(paired_reference[gprof_mask], reference[gprof_mask], rtol=0, atol=0)
        if (gprof[gprof_mask] < 0).any():
            raise ValueError(f'Unexpected negative GPROF prediction: {stamp}')

        prediction_path = cache / f'cnn_{stamp}.nc'
        reused = prediction_path.is_file()
        if reused:
            with xr.open_dataset(prediction_path) as saved:
                if saved.attrs.get('signature') != signature:
                    raise ValueError(f'CNN cache provenance mismatch: {prediction_path}')
                cnn = grid_values(saved, 'surface_precip', target)
        else:
            with torch.inference_mode():
                result = evaluator.evaluate_scene(
                    index=index, retrieval_fn=retrieval, input_data_format='spatial',
                    tile_size=256, overlap=64, batch_size=args.batch_size,
                    track=False, output_path=None,
                )
            cnn = grid_values(result, 'surface_precip', target)
            saved = xr.Dataset({'surface_precip': (DIMS, cnn)},
                               coords={dim: target[dim] for dim in DIMS},
                               attrs={'signature': signature, 'timestamp': stamp})
            temporary = prediction_path.with_suffix('.nc.tmp')
            saved.to_netcdf(temporary, engine='h5netcdf',
                            encoding={'surface_precip': {'zlib': True, 'complevel': 1}})
            temporary.replace(prediction_path)
            del result, saved

        # Every model, including the zero reference, is scored on this exact intersection.
        common = gprof_mask & np.isfinite(cnn)
        n = int(common.sum())
        coverage.append({'timestamp': stamp, 'eligible_radar_pixels': int(eligible.sum()),
                         'gprof_valid_pixels': int(gprof_mask.sum()),
                         'cnn_valid_pixels': int((eligible & np.isfinite(cnn)).sum()),
                         'common_pixels': n})
        if n:
            y = reference[common].astype(np.float64)
            if not np.isfinite(y).all() or (y < 0).any():
                raise ValueError(f'Invalid reference within common mask: {stamp}')
            p = cnn[common].astype(np.float64)
            negative_count += int((p < 0).sum())
            predictions = (p, np.maximum(p, 0), gprof[common].astype(np.float64), np.zeros_like(y))
            n_pixels += n
            reference_sum += float(y.sum())
            for name, prediction in zip(names, predictions):
                error = prediction - y
                totals[name] += [np.abs(error).sum(), np.square(error).sum(), error.sum()]
                for threshold in (0.1, 10.0):
                    observed, detected = y >= threshold, prediction >= threshold
                    counts[name, threshold] += [np.count_nonzero(observed & detected),
                                               np.count_nonzero(~observed & detected),
                                               np.count_nonzero(observed & ~detected)]
        print(f'{index + 1}/600 {stamp}: {n} common pixels ({"cached" if reused else "inferred"})', flush=True)

    if not n_pixels:
        raise RuntimeError('No common valid pixels.')
    metrics = pd.DataFrame([
        {'model': name, 'pixels': n_pixels, 'MAE_mm_h': ae / n_pixels,
         'RMSE_mm_h': np.sqrt(se / n_pixels), 'Bias_mm_h': error / n_pixels,
         'Relative_bias_percent': 100 * ratio(error, reference_sum)}
        for name, (ae, se, error) in totals.items()
    ])
    detection = pd.DataFrame([
        {'model': name, 'threshold_mm_h': threshold, 'TP': tp, 'FP': fp, 'FN': fn,
         'CSI': ratio(tp, tp + fp + fn), 'POD': ratio(tp, tp + fn), 'FAR': ratio(fp, tp + fp)}
        for (name, threshold), (tp, fp, fn) in counts.items()
    ])
    metrics.to_csv(output / 'metrics.csv', index=False)
    detection.to_csv(output / 'rate_detection.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    eligible_total = sum(row['eligible_radar_pixels'] for row in coverage)
    completion = {'complete_test': True, 'scenes': len(scenes), 'signature': signature,
                  'common_pixels': n_pixels, 'eligible_radar_pixels': eligible_total,
                  'common_coverage': ratio(n_pixels, eligible_total),
                  'cnn_negative_fraction_on_common_pixels': negative_count / n_pixels,
                  'note': 'Same scenes, radar reference, grid, mask and rate thresholds; different training data.'}
    (output / 'completion.json').write_text(json.dumps(completion, indent=2), encoding='utf-8')
    print('\n' + metrics.to_string(index=False))
    print('\n' + detection.to_string(index=False))
    print(f'Common coverage: {n_pixels / eligible_total:.6%}')
    print('Saved:', output.resolve())


if __name__ == '__main__':
    main()
