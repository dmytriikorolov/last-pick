"""Losslessly compress raw caches, verifying each file before
removing its
original
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lastpick.data.io import compress_raw_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/raw/opendota/matches"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory does not exist")
    if args.workers < 1:
        parser.error("workers must be positive")
    files = sorted(args.directory.glob("*.json"))
    before = after = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for count, (original, compressed) in enumerate(executor.map(compress_raw_file, files), 1):
            before += original
            after += compressed
            if count % 1000 == 0 or count == len(files):
                print(
                    f"{count}/{len(files)} files; saved {(before - after) / 1024**3:.2f} GiB",
                    flush=True,
                )


if __name__ == "__main__":
    main()
