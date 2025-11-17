"""
Automated Dataset Download Scripts
Downloads all required datasets for deepfake detection
"""
import os
import requests
from tqdm import tqdm
import gdown
import subprocess
from pathlib import Path

class DatasetDownloader:
    """Automated dataset downloader"""
    def __init__(self, base_dir='data/datasets'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest: Path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    def download_asvspoof2019(self):
        """ASVspoof 2019 - Classic deepfake dataset"""
        print("📥 Downloading ASVspoof 2019...")
        print("⚠️  Please visit: https://datashare.ed.ac.uk/handle/10283/3336")
        print("    Manual download required due to license agreement")
        print("    Save to: data/datasets/asvspoof2019/")

    def download_wavefake(self):
        """WaveFake - Vocoder-based fake audio"""
        print("📥 Downloading WaveFake...")
        print("⚠️  Visit: https://zenodo.org/record/5642694")
        print("    Save to: data/datasets/wavefake/")

    def download_librispeech_subset(self):
        """LibriSpeech clean subset"""
        print("📥 Downloading LibriSpeech (dev-clean subset)...")
        url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
        dest = self.base_dir / "librispeech" / "dev-clean.tar.gz"
        dest.parent.mkdir(exist_ok=True)

        if not dest.exists():
            print("Downloading... This may take a while.")
            subprocess.run(['wget', '-O', str(dest), url], check=True)
            print("✅ Downloaded. Extracting...")
            subprocess.run(['tar', '-xzf', str(dest), '-C', str(dest.parent)], check=True)
            print("✅ LibriSpeech ready!")
        else:
            print("✅ LibriSpeech already downloaded")

    def download_commonvoice_sample(self):
        """Mozilla Common Voice sample"""
        print("📥 Common Voice requires account")
        print("    Visit: https://commonvoice.mozilla.org/")
        print("    Download English dataset and save to: data/datasets/commonvoice/")

    def download_for_dataset(self):
        """FoR (Fake-or-Real) dataset"""
        print("📥 FoR Dataset")
        print("    Visit: https://bil.eecs.yorku.ca/datasets/")
        print("    Save to: data/datasets/for/")

    def download_all(self):
        """Download all available datasets"""
        print("=" * 60)
        print("DetectVoice Dataset Downloader")
        print("=" * 60)

        self.download_asvspoof2019()
        self.download_wavefake()
        self.download_librispeech_subset()
        self.download_commonvoice_sample()
        self.download_for_dataset()

        print("\n" + "=" * 60)
        print("✅ Dataset download instructions complete!")
        print("📝 Note: Some datasets require manual download due to licenses")
        print("=" * 60)

if __name__ == '__main__':
    downloader = DatasetDownloader()
    downloader.download_all()
