"""Fetch the official UCI archive and verify the exact Cleveland file used here."""
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/45/heart+disease.zip'
SHA256 = 'a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8'


def verify(raw):
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SHA256:
        raise ValueError(f'Unexpected dataset checksum: {actual}; do not silently substitute another dataset')


def main():
    dest = ROOT / 'data' / 'processed.cleveland.data'
    if dest.exists():
        verify(dest.read_bytes())
        print('Included Cleveland file verified; no download required.')
        return
    request = urllib.request.Request(URL, headers={'User-Agent': 'HeartDiseaseCourseProject/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    names = [n for n in archive.namelist() if Path(n).name == 'processed.cleveland.data']
    if len(names) != 1:
        raise ValueError('Could not identify exactly one processed Cleveland file')
    raw = archive.read(names[0])
    verify(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    print('Downloaded and verified original Cleveland file:', dest.name)


if __name__ == '__main__':
    main()
