"""Native-swath inference; run with .venv_gprof_v8, NOT the SatRain venv."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import hdf5plugin  # noqa: F401
import numpy as np
import torch
import xarray as xr
from gprof_nn import sensors
from gprof_nn.retrieval import GPROFNNInputLoader, calculate_quality_flag_and_pixel_status
from pytorch_retrieve import load_model
from pytorch_retrieve.inference import SequentialInferenceRunner


class SatRainSwathLoader(GPROFNNInputLoader):
    """Adapt observations only; no radar precipitation enters the model."""
    def __init__(self, scenes, signature):
        super().__init__([s['gmi_on_swath'] for s in scenes], config='3d',
                         ancillary_config='NONE', bias_correction=True, robust=False)
        self.scenes = {Path(s['gmi_on_swath']): s for s in scenes}
        self.signature = signature

    def load_input_data(self, path):
        scene = self.scenes[path]
        sensor = sensors.get_sensor('gmi')
        with xr.open_dataset(path, engine='h5netcdf') as ds:
            tb = ds.observations.transpose('channel', 'scan', 'pixel').values.astype('float32')
            times = ds.scan_time
            if times.dims == ('scan',):
                scan_time = times.values.astype('datetime64[ns]')
            elif set(times.dims) == {'scan', 'pixel'}:
                time_values = times.transpose('scan', 'pixel').values.astype('datetime64[ns]')
                valid = ~np.isnat(time_values)
                if not valid.any(axis=1).all():
                    raise ValueError('A scan has no valid observation time.')
                scan_time = time_values[np.arange(tb.shape[1]), valid.argmax(axis=1)]
                # GMI times must describe the same scan, not radar matching times.
                delta = np.abs((time_values - scan_time[:, None]) / np.timedelta64(1, 's'))
                if np.nanmax(delta) > 2:
                    raise ValueError('Within-scan time variation exceeds 2 seconds; inspect schema.')
            else:
                raise ValueError(f'Unsupported scan_time dimensions: {times.dims}')
        if tb.shape[0] != 13 or tb.shape[1] < 100 or tb.shape[2] != 221:
            raise ValueError(f'Expected GMI (13, >=100 scans, 221 pixels), got {tb.shape}. Inspect before adapting.')
        if np.isnat(scan_time).any():
            raise ValueError('Missing GMI scan times.')
        finite = tb[np.isfinite(tb) & (tb > 0)]
        if finite.size == 0 or np.median(finite) < 40:
            raise ValueError('Expected raw brightness temperatures in kelvin, not normalized CNN inputs.')
        tb[(tb < 0) | (tb > 350) | ~np.isfinite(tb)] = np.nan
        shape = tb.shape[1:]
        # These are geolocation coordinates, never precipitation or quality labels.
        with xr.open_dataset(scene['target_on_swath'], engine='h5netcdf') as geo:
            lat = geo.latitude.transpose('scan', 'pixel').values.copy()
            lon = geo.longitude.transpose('scan', 'pixel').values.copy()
        if lat.shape != shape or lon.shape != shape:
            raise ValueError('Observation/geolocation shapes differ.')
        full = np.full((15,) + shape, np.nan, dtype='float32')
        full[sensor.gprof_channel_indices] = tb
        angles = np.full_like(full, np.nan)
        angles[list(sensor.gprof_channels)] = sensor.earth_incidence_angle[:, None, None]
        geolocation = xr.Dataset({'latitude': (('scan', 'pixel'), lat),
                                  'longitude': (('scan', 'pixel'), lon)})
        qflag, status = calculate_quality_flag_and_pixel_status(sensor, full, angles, geolocation)
        inputs = {'brightness_temperatures': torch.from_numpy(full)[None],
                  'earth_incidence_angles': torch.from_numpy(angles)[None],
                  'ancillary_data': torch.full((1, 14) + shape, float('nan'), dtype=torch.float32)}
        aux = {'sensor': sensor, 'ancillary_config': 'NONE', 'output_format': 'NETCDF',
               'pixel_status': status, 'quality_flag': qflag, 'scan_time': scan_time,
               'longitude': lon, 'latitude': lat}
        names = ['total_column_water_vapor', 'two_meter_temperature', 'convective_precipitation',
                 'moisture_convergence', 'leaf_area_index', 'snow_depth', 'orographic_wind',
                 'wind_speed_10m', 'mountain_index', 'land_fraction', 'ice_fraction',
                 'wet_bulb_temperature', 'elevation', 'snow_mask']
        for name in names:
            aux[name] = np.full(shape, np.nan, dtype='float32')
        return inputs, aux

    def finalize_results(self, results, aux, filename, output_path):
        output = super().finalize_results(results, aux, filename, output_path=None)
        output.attrs['experiment_signature'] = self.signature
        output.attrs['adapter'] = 'SatRain Korea native swath; ancillary NONE; nominal GMI angles'
        output.surface_precip.attrs['units'] = 'mm h-1'
        rain = output.surface_precip.values
        valid = np.isfinite(rain) & (rain >= 0)
        if not valid.any():
            raise RuntimeError(f'No valid predictions in {filename}; stopping.')
        print(f'{filename}: shape={rain.shape}, valid={valid.mean():.2%}, max={rain[valid].max():.3f}', flush=True)
        return output, filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('experiments/gprof_korea/manifest.json'))
    parser.add_argument('--model', type=Path, required=True, help='Existing gprof_nn_3d_gmi.pt; no automatic download')
    parser.add_argument('--output', type=Path, default=Path('experiments/gprof_korea/predictions'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--all', action='store_true', help='Require the complete official manifest; default: first scene')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest['domain'] != 'korea' or manifest['split'] != 'testing':
        raise ValueError('Expected Korea testing manifest.')
    scene_count = len(manifest['scenes'])
    expected_count = manifest['official_scene_count']
    if args.all and scene_count != expected_count:
        raise RuntimeError(
            f'Incomplete Korea manifest: {scene_count}/{expected_count} scenes. '
            'Run prepare --all and wait for it to finish successfully before inference.'
        )
    digest = hashlib.sha256()
    with args.model.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    settings = {'model_sha256': digest.hexdigest(), 'gprof_nn': version('gprof_nn'),
                'torch': version('torch'), 'numpy': version('numpy'),
                'pytorch_retrieve': version('pytorch_retrieve'), 'dtype': 'float32',
                'ancillary_config': 'NONE', 'bias_correction': True,
                'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = manifest['scenes'] if args.all else manifest['scenes'][:1]
    pending = []
    for scene in scenes:
        output = args.output / Path(scene['gmi_on_swath']).name
        if output.exists():
            with xr.open_dataset(output) as ds:
                if ds.attrs.get('experiment_signature') != signature:
                    raise RuntimeError(f'Different configuration at {output}; choose a new --output directory.')
                if not ((ds.surface_precip >= 0) & np.isfinite(ds.surface_precip)).any().item():
                    raise RuntimeError(f'Invalid existing output: {output}')
            print('Already processed:', scene['timestamp'])
        else:
            pending.append(scene)
    if pending:
        model = load_model(args.model).eval()
        profiles = ['snow_water_content', 'rain_water_content', 'cloud_water_content', 'latent_heat']
        model.inference_config.retrieval_output = {
            k: v for k, v in model.inference_config.retrieval_output.items() if k not in profiles}
        heads = getattr(model, 'heads', getattr(model, 'outputs', None))
        if heads is not None:
            for name in profiles:
                if name in heads:
                    heads.pop(name)
        (args.output / 'configuration.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        loader = SatRainSwathLoader(pending, signature)
        runner = SequentialInferenceRunner(model, loader, model.inference_config, progress=True, robust=False)
        runner.run(output_path=args.output, device=torch.device(args.device), dtype=torch.float32)
    for scene in scenes:
        if not (args.output / Path(scene['gmi_on_swath']).name).is_file():
            raise RuntimeError(f'Missing prediction: {scene["timestamp"]}')
    print(f'Finished: {len(scenes)} scenes. Predictions: {args.output.resolve()}')


if __name__ == '__main__':
    main()
