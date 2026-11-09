from io import BytesIO
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
from fastapi import HTTPException


def decode_image(data, max_pixels):
    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.width * probe.height > max_pixels:
                raise HTTPException(413, "Imagem excede o limite de pixels.")
            probe.verify()
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(400, "Arquivo de imagem inválido.") from exc
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Formato de imagem não suportado.")
    return image


async def bounded_read(file, limit):
    chunks = []
    size = 0
    try:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, "Arquivo excede o limite de upload.")
            chunks.append(chunk)
        if not size:
            raise HTTPException(400, "Arquivo vazio.")
        return b"".join(chunks)
    finally:
        await file.close()
