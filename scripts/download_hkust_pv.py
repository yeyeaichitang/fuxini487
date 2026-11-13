"""Download and verify the public HKUST rooftop PV release from Dryad."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/raw/public_datasets/hkust_rooftop_pv_2021_2023'
API = 'https://datadryad.org'
DATASET = '/api/v2/datasets/doi%3A10.5061%2Fdryad.m37pvmd99'


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'FuXi-project-public-dataset-downloader/1.0'
    }), timeout=90)


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    metadata = json.load(request(API + DATASET))
    version = metadata['_links']['stash:version']['href']
    listing = json.load(request(API + version + '/files'))
    save_json(DEST / 'source_metadata.json', metadata)
    save_json(DEST / 'source_files.json', listing)
    # Dryad file bytes require authentication on this host. Its public Zenodo
    # archive is the April 2024 release, not the September Dryad release.
    archive_metadata = json.load(request('https://zenodo.org/api/records/10909062'))
    save_json(DEST / 'zenodo_metadata.json', archive_metadata)
    records = []
    for entry in archive_metadata['files']:
        name = entry['key']
        if Path(name).name != name:
            raise ValueError('Unexpected source filename')
        target = DEST / name
        url = entry['links']['self']
        algorithm, expected = entry['checksum'].split(':', 1)
        if not (target.exists() and target.stat().st_size == entry['size']
                and hashlib.file_digest(target.open('rb'), algorithm).hexdigest() == expected):
            partial = target.with_suffix(target.suffix + '.part')
            for attempt in range(3):
                try:
                    digest = hashlib.new(algorithm)
                    size = 0
                    last_report = time.monotonic()
                    with request(url) as response, partial.open('wb') as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                            if time.monotonic() - last_report > 15:
                                print(f'{name}: {size}/{entry["size"]} bytes', flush=True)
                                last_report = time.monotonic()
                    if size != entry['size'] or digest.hexdigest() != expected:
                        raise ValueError('Source size or checksum mismatch')
                    partial.replace(target)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    print(f'Retrying {name}, attempt {attempt + 2}', flush=True)
        with target.open('rb') as handle:
            sha256 = hashlib.file_digest(handle, 'sha256').hexdigest()
        records.append({'file': name, 'bytes': target.stat().st_size,
                        'source_checksum': entry['checksum'], 'sha256': sha256,
                        'download_url': url, 'verified': True})
        print(f'Verified {name}', flush=True)
    extracted = DEST / 'extracted'
    extracted.mkdir(exist_ok=True)
    inventory = []
    with zipfile.ZipFile(DEST / 'dataset.zip') as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP CRC failure: {bad}')
        for entry in archive.infolist():
            rel = PurePosixPath(entry.filename.replace('\\', '/'))
            output = (extracted / str(rel)).resolve()
            if rel.is_absolute() or not output.is_relative_to(extracted.resolve()):
                raise ValueError('Unsafe archive path')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlink unsupported')
            if entry.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                with archive.open(entry) as source, output.open('xb') as dest:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
            # Verify extracted bytes, including files left by interrupted runs.
            import zlib
            crc = 0
            with output.open('rb') as handle:
                while chunk := handle.read(1024 * 1024):
                    crc = zlib.crc32(chunk, crc)
            if output.stat().st_size != entry.file_size or crc != entry.CRC:
                raise ValueError(f'Extracted file verification failed: {entry.filename}')
            inventory.append({'path': output.relative_to(DEST).as_posix(),
                              'bytes': entry.file_size, 'crc32': f'{crc:08x}'})
    save_json(DEST / 'file_inventory.json', inventory)
    report = {'dataset_doi': metadata['identifier'], 'latest_dryad_version_reference': version,
              'download_source': 'https://zenodo.org/records/10909062',
              'downloaded_release_date': archive_metadata['metadata']['publication_date'],
              'release_note': 'April 2024 public archive; not the September 2024 Dryad revision',
              'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
              'license': metadata['license'], 'files': records,
              'extracted_files': len(inventory),
              'extracted_bytes': sum(x['bytes'] for x in inventory),
              'zip_crc_verified': True, 'extracted_crc_verified': True,
              'training_ready': False}
    save_json(DEST / 'download_manifest.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
