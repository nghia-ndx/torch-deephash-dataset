import os
from threading import Thread
from typing import IO, List, Optional

import requests
from tqdm.auto import tqdm


def download_with_progress_bar(
    url: str, save_buffer: IO[bytes], desc: Optional[str] = None, keep_pbar=True
):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    chunk_size = 1024
    with tqdm(
        desc=desc,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=chunk_size,
        leave=keep_pbar,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = save_buffer.write(data)
            pbar.update(size)


def download_file_list(urls: List[str], save_folder: str):
    def download_and_write_to_file(url: str):
        file_name = url.split('/')[-1]
        with open(os.path.join(save_folder, file_name), 'wb') as file:
            download_with_progress_bar(url, file, file_name, keep_pbar=False)

    threads = [Thread(target=download_and_write_to_file, args=(url,)) for url in urls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
