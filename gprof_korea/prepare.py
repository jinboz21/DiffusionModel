"""Download matched Korea scenes. Run with the SatRain (.venv) interpreter."""
import argparse
import json
from pathlib import Path, PurePosixPath

import hdf5plugin  # noqa: F401: register NetCDF compression filters
import xarray as xr
from satrain.data import download_files, get_data_url, get_files_in_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, default=Path('data/satrain_store'))
    parser.add_argument('--manifest', type=Path, default=Path('experiments/gprof_korea/manifest.json'))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--limit', type=int, default=1)
    group.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    root = args.data_path.resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog = get_files_in_dataset('satrain')['gmi']['testing']['korea']
    maps = {}
    for geometry, source in [('on_swath', 'gmi'), ('on_swath', 'target'), ('gridded', 'target')]:
        key = f'{source}_{geometry}'
        mapping = {}
        for filename in catalog[geometry][source]:
            filename = str(filename).replace('\\', '/')
            path = PurePosixPath(filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unexpected remote path: {filename}')
            stamp = path.stem.rsplit('_', 1)[-1]
            if len(stamp) != 14 or not stamp.isdigit() or stamp in mapping:
                raise ValueError(f'Invalid or duplicate scene identifier: {filename}')
            mapping[stamp] = filename
        maps[key] = mapping
    timestamps = set(next(iter(maps.values())))
    if not timestamps or any(set(mapping) != timestamps for mapping in maps.values()):
        raise RuntimeError('Remote sources do not contain identical scene IDs; refusing partial pairing.')
    selected = sorted(timestamps)
    if not args.all:
        selected = selected[:args.limit]
    print(f'Official Korea scenes: {len(timestamps)}; selected: {len(selected)}', flush=True)
    relative_paths = [mapping[stamp] for stamp in selected for mapping in maps.values()]
    missing = [p for p in relative_paths if not (root / p).is_file() or (root / p).stat().st_size == 0]
    if missing:
        downloaded = download_files(get_data_url('satrain'), missing, root)
        if set(downloaded) != set(missing):
            raise RuntimeError('Download incomplete. Run this command again before proceeding.')
    scenes = []
    for stamp in selected:
        scene = {'timestamp': stamp}
        for key, mapping in maps.items():
            scene[key] = str(root / mapping[stamp])
            # Opening each file catches incomplete downloads before inference.
            with xr.open_dataset(scene[key], engine='h5netcdf') as ds:
                required = ['observations', 'scan_time'] if key == 'gmi_on_swath' else ['surface_precip']
                if key == 'target_on_swath':
                    required += ['latitude', 'longitude']
                if key == 'target_gridded':
                    required += ['scan_index', 'pixel_index']
                for variable in required:
                    if variable not in ds:
                        raise RuntimeError(f'{key} missing {variable}: {scene[key]}')
                if stamp == selected[0]:
                    print(f'{key}: {dict(ds.sizes)}')
                    if key == 'gmi_on_swath':
                        print('Observation attributes:', ds.observations.attrs)
        scenes.append(scene)
    document = {'domain': 'korea', 'split': 'testing', 'base_sensor': 'gmi',
                'data_path': str(root), 'official_scene_count': len(timestamps), 'scenes': scenes}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = args.manifest.with_suffix('.json.tmp')
    temporary_manifest.write_text(json.dumps(document, indent=2), encoding='utf-8')
    temporary_manifest.replace(args.manifest)
    size = sum((root / p).stat().st_size for p in relative_paths)
    print(f'Selected files: {len(relative_paths)}; stored size: {size / 1024**3:.3f} GiB')
    print('Manifest:', args.manifest.resolve())
    print(f'PREPARATION COMPLETE: {len(scenes)}/{len(timestamps)} official scenes indexed.')


if __name__ == '__main__':
    main()
