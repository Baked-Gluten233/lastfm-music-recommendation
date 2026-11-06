"""Official download with checksum verification; no account or API key required."""
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from musicrec.data import ARCHIVE_SHA256, DATA_URL, sha256

def main():
    destination = ROOT / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "hetrec2011-lastfm-2k.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        try:
            with urlopen(DATA_URL, timeout=60) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if sha256(temporary) != ARCHIVE_SHA256:
                raise ValueError("Archive checksum changed; do not use without source review")
            temporary.replace(archive)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive checksum mismatch")
    target = destination / "hetrec2011-lastfm-2k"
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = (target / member.filename).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive member")
        zipped.extractall(target)
    print("Verified official dataset:", target)

if __name__ == "__main__":
    main()
