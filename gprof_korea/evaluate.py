"""Score cached predictions with SatRain's official Korea grid and quality mask."""
import argparse
import json
from pathlib import Path

import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from satrain.evaluation import Evaluator
from satrain.input import GMI
from satrain.target import TargetConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('experiments/gprof_korea/manifest.json'))
    parser.add_argument('--predictions', type=Path, default=Path('experiments/gprof_korea/predictions'))
    parser.add_argument('--output', type=Path, default=Path('experiments/gprof_korea/evaluation'))
    parser.add_argument('--all', action='store_true', help='Require and evaluate the complete official manifest')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest['domain'] != 'korea' or manifest['split'] != 'testing':
        raise ValueError('Expected Korea testing manifest.')
    scenes = manifest['scenes'] if args.all else manifest['scenes'][:1]
    if args.all and len(scenes) != manifest['official_scene_count']:
        raise RuntimeError('Run prepare --all and infer --all before full evaluation.')
    for scene in scenes:
        path = args.predictions / Path(scene['gmi_on_swath']).name
        if not path.is_file():
            raise FileNotFoundError(path)
    target_config = TargetConfig()
    evaluator = Evaluator(base_sensor='gmi', geometry='on_swath', domain='korea',
                          retrieval_input=[GMI(normalize='minmax', nan=-1.5, include_angles=False)],
                          target_config=target_config, data_path=Path(manifest['data_path']), download=False)
    # Unused sources may have been downloaded for other experiments with different counts.
    for source in ['gmi', 'atms', 'ancillary', 'geo', 'geo_t', 'geo_ir', 'geo_ir_t']:
        for geometry in ['gridded', 'on_swath']:
            attribute = f'{source}_{geometry}'
            if hasattr(evaluator, attribute):
                delattr(evaluator, attribute)
    # Override discovery with the explicit paired manifest: never rely on directory ordering.
    evaluator.target_gridded = [Path(s['target_gridded']) for s in scenes]
    evaluator.target_on_swath = [Path(s['target_on_swath']) for s in scenes]
    evaluator.gmi_on_swath = [Path(s['gmi_on_swath']) for s in scenes]
    # This callback supplies only rates and rate-threshold detections, not probabilities.
    evaluator.prob_precip_detection_metrics = []
    evaluator.prob_heavy_precip_detection_metrics = []
    args.output.mkdir(parents=True, exist_ok=True)
    grid_dir = args.output / 'gridded'
    grid_dir.mkdir(exist_ok=True)
    coverage = []
    totals = dict(n=0, ae=0., se=0., error=0., zero_ae=0., zero_se=0., target_sum=0.)
    confusion = {0.1: np.zeros(3, dtype=np.int64), 10.: np.zeros(3, dtype=np.int64)}
    signature = None
    for index, scene in enumerate(scenes):
        path = args.predictions / Path(scene['gmi_on_swath']).name
        with xr.open_dataset(path) as ds:
            current = ds.attrs.get('experiment_signature')
            if current is None or (signature is not None and signature != current):
                raise RuntimeError('Missing or inconsistent prediction provenance.')
            signature = current
            rain = ds.surface_precip.transpose('scans', 'pixels').values.copy()
        rain[~np.isfinite(rain) | (rain < 0)] = np.nan
        with xr.open_dataset(scene['target_gridded'], engine='h5netcdf') as target:
            si, pi = target.scan_index.values, target.pixel_index.values
            mapped = pi >= 0
            if (not np.isfinite(si[mapped]).all() or not np.isfinite(pi[mapped]).all()
                    or (si[mapped] < 0).any() or (si[mapped] >= rain.shape[0]).any()
                    or (pi[mapped] >= rain.shape[1]).any()
                    or (si[mapped] != np.floor(si[mapped])).any()
                    or (pi[mapped] != np.floor(pi[mapped])).any()):
                raise ValueError(f'Invalid grid-to-swath mapping: {scene["timestamp"]}')
            reference = target.surface_precip.values.copy()
            eligible = mapped & ~target_config.get_mask(target)

        def cached_prediction(input_data):
            if (input_data.sizes['scan'], input_data.sizes['pixel']) != rain.shape:
                raise ValueError('Cached predictions do not match the SatRain scene.')
            dims = ('scan', 'pixel')
            return xr.Dataset({'surface_precip': (dims, rain),
                               'precip_flag': (dims, rain >= target_config.precip_threshold),
                               'heavy_precip_flag': (dims, rain >= target_config.heavy_precip_threshold)})

        # SatRain itself applies scan_index/pixel_index and tracks its official metrics.
        result = evaluator.evaluate_scene(index, tile_size=None, overlap=0, batch_size=None,
                                          retrieval_fn=cached_prediction, input_data_format='spatial',
                                          track=True, output_path=None)
        predicted = result.surface_precip.values
        if predicted.shape != reference.shape:
            raise ValueError('Evaluation grid differs from radar grid.')
        valid = eligible & np.isfinite(predicted)
        n = int(valid.sum())
        coverage.append({'timestamp': scene['timestamp'], 'eligible_radar_pixels': int(eligible.sum()),
                         'evaluated_pixels': n, 'missing_prediction_pixels': int((eligible & ~valid).sum())})
        if n:
            p, y = predicted[valid].astype('float64'), reference[valid].astype('float64')
            error = p - y
            totals['n'] += n
            totals['ae'] += np.abs(error).sum()
            totals['se'] += np.square(error).sum()
            totals['error'] += error.sum()
            totals['zero_ae'] += np.abs(y).sum()
            totals['zero_se'] += np.square(y).sum()
            totals['target_sum'] += y.sum()
            for threshold, counts in confusion.items():
                observed, detected = y >= threshold, p >= threshold
                counts += [np.count_nonzero(observed & detected),
                           np.count_nonzero(~observed & detected), np.count_nonzero(observed & ~detected)]
        # Explicit mask allows CNN evaluation on exactly the same pixels later.
        result['common_valid_mask'] = (result.surface_precip.dims, valid.astype('uint8'))
        result.attrs['experiment_signature'] = signature
        result.to_netcdf(grid_dir / f'paired_{scene["timestamp"]}.nc')
        print(f'Evaluated {index + 1}/{len(scenes)}: {n} pixels', flush=True)
    if totals['n'] == 0:
        raise RuntimeError('No valid paired radar/prediction pixels.')
    for name, getter in [('official_quantification', evaluator.get_precip_quantification_results),
                         ('official_detection', evaluator.get_precip_detection_results),
                         ('official_heavy_detection', evaluator.get_heavy_precip_detection_results)]:
        getter(name='GPROF-NN V8 NONE', include_baselines=False).to_csv(args.output / f'{name}.csv')
    n = totals['n']
    metrics = [{'model': 'GPROF-NN V8 NONE', 'MAE': totals['ae']/n,
                'RMSE': np.sqrt(totals['se']/n), 'Bias': totals['error']/n},
               {'model': 'Always_zero_same_pixels', 'MAE': totals['zero_ae']/n,
                'RMSE': np.sqrt(totals['zero_se']/n), 'Bias': -totals['target_sum']/n}]
    table = pd.DataFrame(metrics)
    table.to_csv(args.output / 'metrics.csv', index=False)
    detections = []
    for threshold, (tp, fp, fn) in confusion.items():
        detections.append({'threshold_mm_h': threshold, 'TP': tp, 'FP': fp, 'FN': fn,
                           'CSI': tp/(tp+fp+fn) if tp+fp+fn else np.nan,
                           'POD': tp/(tp+fn) if tp+fn else np.nan,
                           'FAR': fp/(tp+fp) if tp+fp else np.nan})
    pd.DataFrame(detections).to_csv(args.output / 'rate_detection.csv', index=False)
    pd.DataFrame(coverage).to_csv(args.output / 'coverage.csv', index=False)
    eligible_count = sum(row['eligible_radar_pixels'] for row in coverage)
    summary = {'scenes': len(scenes), 'official_scene_count': manifest['official_scene_count'],
               'complete_test': len(scenes) == manifest['official_scene_count'],
               'eligible_radar_pixels': eligible_count, 'evaluated_pixels': n,
               'prediction_coverage': n/eligible_count, 'experiment_signature': signature,
               'target_config': vars(target_config),
               'comparison_note': 'CNN must use these same scene IDs and common_valid_mask pixels.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(table.to_string(index=False))
    print(f'Prediction coverage on eligible radar pixels: {n/eligible_count:.2%}')
    print('This is a full Korea run.' if summary['complete_test'] else 'SMOKE ONLY: not a full Korea benchmark.')
    print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
