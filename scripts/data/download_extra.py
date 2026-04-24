"""
Télécharge les sources de passwords additionnelles (SecLists, etc.)
À lancer une fois sur le cluster avant prepare_dataset.py.

Usage: python scripts/data/download_extra.py
"""

import os, urllib.request, tarfile, time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTRA_DIR = os.path.join(BASE_DIR, 'data', 'extra')

SOURCES = [
    {
        'name': 'pwdb_top10M.txt',
        'url':  'https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/Pwdb_top-10000000.txt',
        'desc': 'Top 10M passwords (multi-breach compilation)',
    },
    {
        'name': 'rockyou-withcount.txt.tar.gz',
        'url':  'https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Leaked-Databases/rockyou-withcount.txt.tar.gz',
        'desc': 'RockYou avec fréquences (pour freq-weighting)',
        'extract': True,
    },
    {
        'name': '000webhost.txt',
        'url':  'https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Leaked-Databases/000webhost.txt',
        'desc': '000webhost leak (~720k passwords)',
    },
    {
        'name': 'phpbb.txt',
        'url':  'https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Leaked-Databases/phpbb.txt',
        'desc': 'phpbb forum leak (~184k passwords)',
    },
]


def download(url, dest):
    t0 = time.time()
    def progress(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            mb  = count * block / 1024 / 1024
            print(f"   {pct:3d}% ({mb:.1f} MB)", end='\r')
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"   OK ({size_mb:.1f} MB en {time.time()-t0:.0f}s)")


def main():
    os.makedirs(EXTRA_DIR, exist_ok=True)
    print("=" * 60)
    print("TÉLÉCHARGEMENT DES SOURCES PASSWORDS")
    print("=" * 60)

    for src in SOURCES:
        dest = os.path.join(EXTRA_DIR, src['name'])
        print(f"\n{src['name']} — {src['desc']}")
        if os.path.exists(dest):
            print(f"   Déjà présent ({os.path.getsize(dest)//1024//1024} MB), skip")
        else:
            try:
                download(src['url'], dest)
            except Exception as e:
                print(f"   ERREUR: {e}")
                continue

        # Extraire si tar.gz
        if src.get('extract') and dest.endswith('.tar.gz'):
            extracted_name = src['name'].replace('.tar.gz', '')
            extracted_path = os.path.join(EXTRA_DIR, extracted_name)
            if not os.path.exists(extracted_path):
                print(f"   Extraction...")
                with tarfile.open(dest, 'r:gz') as tar:
                    tar.extractall(EXTRA_DIR)
                print(f"   Extrait: {extracted_name}")

    print(f"\n{'='*60}")
    print(f"Fichiers dans {EXTRA_DIR}:")
    for f in sorted(os.listdir(EXTRA_DIR)):
        size_mb = os.path.getsize(os.path.join(EXTRA_DIR, f)) / 1024 / 1024
        print(f"   {f:40s} {size_mb:8.1f} MB")
    print(f"{'='*60}")
    print("\nProchaine étape: python scripts/data/prepare_dataset.py --freq-weight")


if __name__ == '__main__':
    main()
