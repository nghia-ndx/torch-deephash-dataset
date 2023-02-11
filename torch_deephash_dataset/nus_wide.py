import csv
import os
import shutil

from tqdm.auto import tqdm

from .base import BaseDeepHashDataset
from .utils.download import download_file_list, download_with_progress_bar
from .utils.log import logger

_REPO_BASE_URL = 'https://raw.githubusercontent.com/nghia-ndx/deephash-nus-wide/main'
_ARCHIVE_DIR_URL = f'{_REPO_BASE_URL}/images'
_IMAGE_ARCHIVES = [f'images_{i:02d}.zip' for i in range(56)]

_ARCHIVE_SAVE_DIR = 'archives'
_EXTRACTED_IMAGE_SAVE_DIR = 'images'


class NUSWIDEDataset(BaseDeepHashDataset):
    @property
    def archive_save_dir(self):
        return self.get_full_path(_ARCHIVE_SAVE_DIR)

    @property
    def extracted_image_save_dir(self):
        return self.get_full_path(_EXTRACTED_IMAGE_SAVE_DIR)

    def download_dataset(self):
        os.makedirs(self.archive_save_dir, exist_ok=True)
        os.makedirs(self.extracted_image_save_dir, exist_ok=True)

        logger.info('Cloning dataset from repo...')

        download_batches = [
            _IMAGE_ARCHIVES[i : i + 4] for i in range(0, len(_IMAGE_ARCHIVES), 4)
        ]
        for batch in tqdm(download_batches, desc='Downloading in batches'):
            download_file_list(
                [f'{_ARCHIVE_DIR_URL}/{archive}' for archive in batch],
                save_folder=self.archive_save_dir,
            )

        for split in self.allowed_dataset_splits:
            with open(os.path.join(self.root, f'{split}.csv'), 'wb') as file:
                download_with_progress_bar(
                    f'{_REPO_BASE_URL}/{split}.csv', file, f'{split}.csv'
                )

        logger.info('Extracting archives')
        for archive in tqdm(_IMAGE_ARCHIVES, desc='Extracting'):
            shutil.unpack_archive(
                os.path.join(self.archive_save_dir, archive),
                self.extracted_image_save_dir,
            )

    def get_split_iterator(self, split: str):
        with open(self.get_full_path(f'{split}.csv'), newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            for path, *label_vals in csv_reader:
                yield path, [int(val) for val in label_vals]

    def is_dataset_existed(self):
        for split in self.allowed_dataset_splits:
            if not os.path.exists(self.get_full_path(f'{split}.csv')):
                return False

            for img_path, *_ in self.get_split_iterator(split):
                img_path = os.path.join(self.root, img_path)
                if not os.path.exists(img_path):
                    return False

        return True
