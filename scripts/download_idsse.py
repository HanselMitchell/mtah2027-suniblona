"""Step 2: download the 7 IDSSE (Sportec / DFL) matches and convert them.

IDSSE: Bassek et al. (2025), Scientific Data 12:195, CC-BY 4.0.
Public data about OTHER games -- allowed by the competition rules.

    python scripts/download_idsse.py              # all 7 matches
    python scripts/download_idsse.py J03WMX       # just one

For each match it needs two XML files in data/idsse/raw/:
    <match_id>_meta.xml        match information
    <match_id>_tracking.xml    positions, ball status, possession

The script tries to download them itself. Figshare sometimes refuses
non-browser downloads; in that case it prints the links. Open each link in
your browser, and save the file under the name shown into data/idsse/raw/,
then run the script again -- it picks up files that are already there.

Writes data/idsse/<match_id>.npz and prints a summary per match.
"""

import os
import shutil
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.tracking_io import dataset_to_arrays, summarize  # noqa: E402

# figshare file ids, as published by kloppy (kloppy/_providers/sportec.py)
FILES = {
    "J03WMX": {"meta": 51643475, "tracking": 51643514},
    "J03WN1": {"meta": 51643472, "tracking": 51643517},
    "J03WOH": {"meta": 51643478, "tracking": 51643520},
    "J03WOY": {"meta": 51643481, "tracking": 51643523},
    "J03WPY": {"meta": 51643487, "tracking": 51643526},
    "J03WQQ": {"meta": 51643484, "tracking": 51643529},
    "J03WR9": {"meta": 51643490, "tracking": 51643532},
}
URL = "https://springernature.figshare.com/ndownloader/files/{}"
RAW_DIR = os.path.join(ROOT, "data", "idsse", "raw")
OUT_DIR = os.path.join(ROOT, "data", "idsse")
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
    "Accept": "*/*",
}


def looks_like_xml(path):
    """True if the file exists, is not tiny, and starts like an XML document."""
    if not os.path.isfile(path) or os.path.getsize(path) < 1000:
        return False
    with open(path, "rb") as handle:
        head = handle.read(200).lstrip(b"\xef\xbb\xbf").lstrip()
    return head.startswith(b"<")


def try_download(file_id, path):
    """Download to path. Returns True on success, False if figshare refused."""
    tmp = path + ".part"
    try:
        request = urllib.request.Request(URL.format(file_id), headers=HEADERS)
        with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except Exception as error:  # network errors, HTTP errors
        print("    download failed: {}".format(error))
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    if looks_like_xml(tmp):
        os.replace(tmp, path)
        return True
    os.remove(tmp)
    print("    figshare returned something that is not XML (bot protection)")
    return False


def unpack_zips():
    """Extract the XML files of any zip dropped into data/idsse/raw/ (e.g. figshare
    'Download all'). Nested zips are handled too. Already-extracted files are kept."""
    import zipfile

    changed = True
    while changed:
        changed = False
        for name in os.listdir(RAW_DIR):
            if not name.lower().endswith(".zip"):
                continue
            path = os.path.join(RAW_DIR, name)
            marker = path + ".extracted"
            if os.path.exists(marker):
                continue
            print("  extracting {} ...".format(name), flush=True)
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    base = os.path.basename(member)
                    if not base.lower().endswith((".xml", ".zip")):
                        continue
                    target = os.path.join(RAW_DIR, base)
                    if not os.path.exists(target):
                        with archive.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=1 << 20)
                        changed = True
            open(marker, "w").close()


def find_local(match_id, kind):
    """A file in data/idsse/raw/ for this match and kind, under any name.

    Accepts our own names (<id>_meta.xml / <id>_tracking.xml) and the original
    DFL names, e.g. ..._matchinformation_..._DFL-MAT-J03WMX.xml (meta) and
    ..._positions_..._DFL-MAT-J03WMX.xml (tracking).
    """
    own = os.path.join(RAW_DIR, "{}_{}.xml".format(match_id, kind))
    if looks_like_xml(own):
        return own
    keyword = "matchinformation" if kind == "meta" else "position"
    for name in sorted(os.listdir(RAW_DIR)):
        low = name.lower()
        if match_id.lower() in low and keyword in low and low.endswith(".xml"):
            path = os.path.join(RAW_DIR, name)
            if looks_like_xml(path):
                return path
    return None


def ensure_files(match_id):
    """Make sure both XML files exist locally. Returns (meta, tracking) or None."""
    paths = {}
    missing = []
    for kind in ("meta", "tracking"):
        found = find_local(match_id, kind)
        path = found or os.path.join(RAW_DIR, "{}_{}.xml".format(match_id, kind))
        paths[kind] = path
        if found:
            continue
        print("  {} {}: downloading ...".format(match_id, kind), flush=True)
        if not try_download(FILES[match_id][kind], path):
            missing.append((kind, path))
    if missing:
        print("\n  Could not download automatically. Open these links in your browser")
        print("  and save the files into {}".format(RAW_DIR))
        print("  (any file name is fine -- the original DFL names are recognised):\n")
        for kind, path in missing:
            print("    {} ({})".format(URL.format(FILES[match_id][kind]), kind))
        print()
        return None
    return paths["meta"], paths["tracking"]


def main(match_ids):
    from kloppy import sportec

    os.makedirs(RAW_DIR, exist_ok=True)
    unpack_zips()
    for match_id in match_ids:
        if match_id not in FILES:
            print("unknown match id {}; choose from {}".format(match_id, list(FILES)))
            continue
        out = os.path.join(OUT_DIR, match_id + ".npz")
        if os.path.exists(out):
            print("{} already converted, skipping".format(match_id))
            continue
        files = ensure_files(match_id)
        if files is None:
            continue
        start = time.time()
        print("{}: parsing (a few minutes) ...".format(match_id), flush=True)
        dataset = sportec.load_tracking(meta_data=files[0], raw_data=files[1],
                                        only_alive=False)
        arrays = dataset_to_arrays(dataset)
        np.savez_compressed(out, **arrays)
        print(summarize(arrays, match_id))
        print("  saved {} ({:.0f} MB) in {:.0f} s\n".format(
            os.path.relpath(out, ROOT), os.path.getsize(out) / 1e6, time.time() - start),
            flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(FILES))
