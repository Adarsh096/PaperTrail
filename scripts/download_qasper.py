"""Download the official v0.3 archive; retain and verify only training data."""
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
TRAIN_SHA256 = "9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4"


def main():
    target = Path("data/qasper-train-v0.3.json")
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read(100_000_001)
    if len(data) > 100_000_000:
        raise ValueError("Archive exceeds 100 MB download limit")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = archive.getmember("qasper-train-v0.3.json")
        if member.size > 100_000_000:
            raise ValueError("Training JSON exceeds size limit")
        train = archive.extractfile(member).read()
    if hashlib.sha256(train).hexdigest() != TRAIN_SHA256:
        raise ValueError("Unexpected QASPER training content; refusing replacement")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(train)
    print(f"Verified training JSON saved to {target}")


if __name__ == "__main__":
    main()
