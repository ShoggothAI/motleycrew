import mimetypes
import os
from typing import Optional

import requests

from motleycrew.common import logger, utils as motley_utils


def download_image(url: str, file_path: str) -> Optional[str]:
    response = requests.get(url, stream=True)
    if response.status_code == requests.codes.ok:
        try:
            content_type = response.headers.get("content-type")
            extension = mimetypes.guess_extension(content_type)
        except Exception as e:
            logger.error("Failed to guess content type: %s", e)
            extension = None

        if not extension:
            extension = ".png"

        file_path_with_extension = file_path + extension
        logger.info("Downloading image %s to %s", url, file_path_with_extension)

        with open(file_path_with_extension, "wb") as f:
            for chunk in response:
                f.write(chunk)

        return file_path_with_extension
    else:
        logger.error("Failed to download image. Status code: %s", response.status_code)


def download_url_to_directory(url: str, images_directory: str, file_name_length: int = 8) -> str:
    os.makedirs(images_directory, exist_ok=True)
    file_name = motley_utils.generate_hex_hash(url, length=file_name_length)
    file_path = os.path.join(images_directory, file_name)

    file_path_with_extension = download_image(url=url, file_path=file_path).replace(os.sep, "/")
    return file_path_with_extension
