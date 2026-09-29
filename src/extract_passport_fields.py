"""Извлекаем поля с выровненной третьей страницы макета."""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from rapidocr import (
    EngineType,
    LangDet,
    LangRec,
    ModelType,
    OCRVersion,
    RapidOCR,
)
from rapidocr.utils.output import RapidOCROutput

if __package__:
    from .passport_page_detector import find_page, straighten_candidate
else:
    from passport_page_detector import find_page, straighten_candidate


# Границы полей в долях от размера выровненной страницы.
FIELD_BOXES = {
    "surname": (0.48, 0.10, 0.93, 0.17),
    "first_name": (0.48, 0.25, 0.93, 0.32),
    "patronymic": (0.46, 0.34, 0.93, 0.41),
    "sex": (0.36, 0.41, 0.50, 0.47),
    "birth_date": (0.59, 0.41, 0.90, 0.47),
    # Место рождения может занимать несколько строк.
    "birth_place": (0.48, 0.49, 0.93, 0.70),
}
PASSPORT_NUMBER_BOX = (0.92, 0.23, 0.995, 0.74)
PASSPORT_NUMBER_TEXT_BOX = (0.92, 0.49, 0.995, 0.74)

_LATIN_LOOKALIKES = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "J": "Л",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
    }
)


def create_ocr() -> RapidOCR:
    """Создаём PP-OCRv5 с восточнославянским распознавателем и ONNX Runtime."""
    return RapidOCR(
        params={
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.ESLAV,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Global.use_cls": False,
        }
    )


def crop_fraction(
    image: np.ndarray, box: tuple[float, float, float, float]
) -> np.ndarray:
    """Вырезаем область, заданную долями от размера страницы."""
    height, width = image.shape[:2]
    left, top, right, bottom = box
    x1, x2 = round(left * width), round(right * width)
    y1, y2 = round(top * height), round(bottom * height)
    return image[y1:y2, x1:x2]


def normalize_russian_text(text: str, *, birth_place: bool = False) -> str:
    """Приводим текст к кириллице и исправляем частые ошибки OCR."""
    value = text.upper().translate(_LATIN_LOOKALIKES)
    if any(character.isalpha() for character in value):
        value = value.replace("3", "З")
    # В русских полях латинских букв быть не должно.
    value = re.sub(r"[A-Z]", "", value)
    if birth_place:
        value = re.sub(r"^ФО[ПР](?=\.)", "ГОР", value)
        value = re.sub(r"^ФОРОД", "ГОРОД", value)
        value = re.sub(r"^(ГОР\.)(?=\S)", r"\1 ", value)
        value = re.sub(r"^(ГОРОД)(?=[А-Я])", r"\1 ", value)
    return value


def _inside_box(
    center_x: float,
    center_y: float,
    box: tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = box
    return left <= center_x <= right and top <= center_y <= bottom


def extract_fields(page: np.ndarray, ocr: RapidOCR) -> dict[str, str]:
    """Распознаём поля на уже выровненной странице."""
    fields = {name: "" for name in FIELD_BOXES}
    grouped_text = {name: [] for name in FIELD_BOXES}

    # Распознаём всю страницу: на маленьких вырезках терялись пол и дата рождения.
    result = ocr(page)
    if (
        isinstance(result, RapidOCROutput)
        and result.boxes is not None
        and result.txts is not None
    ):
        height, width = page.shape[:2]
        for polygon, text in zip(result.boxes, result.txts):
            points = np.asarray(polygon, dtype=np.float32)
            center_x = float(points[:, 0].mean()) / width
            center_y = float(points[:, 1].mean()) / height
            for name, box in FIELD_BOXES.items():
                if _inside_box(center_x, center_y, box):
                    grouped_text[name].append((center_y, center_x, str(text)))

    for name, lines in grouped_text.items():
        lines.sort(key=lambda line: (line[0], line[1]))
        text = " ".join(line[2].strip() for line in lines if line[2].strip())
        fields[name] = normalize_russian_text(
            text, birth_place=name == "birth_place"
        )

    # Поворачиваем полосу с номером и распознаём обе строки отдельно,
    # чтобы OCR не сливал соседние цифры.
    strip = crop_fraction(page, PASSPORT_NUMBER_BOX)
    strip = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    split_x = round(strip.shape[1] * 0.56)
    series_crop = strip[:, :split_x]
    number_crop = strip[:, split_x:]
    number_result = ocr.recognize_txt([series_crop, number_crop])
    recognized_texts = number_result.txts or ()
    series_text = recognized_texts[0] if len(recognized_texts) > 0 else ""
    number_text = recognized_texts[1] if len(recognized_texts) > 1 else ""
    series_digits = re.sub(r"\D", "", series_text)
    number_digits = re.sub(r"\D", "", number_text)

    # Если отдельный фрагмент распознался не полностью, берём цифры
    # из результата распознавания всей страницы.
    if len(number_digits) != 6 and isinstance(result, RapidOCROutput):
        number_box_items = []
        if result.boxes is not None and result.txts is not None:
            height, width = page.shape[:2]
            for polygon, text in zip(result.boxes, result.txts):
                points = np.asarray(polygon, dtype=np.float32)
                center_x = float(points[:, 0].mean()) / width
                center_y = float(points[:, 1].mean()) / height
                if _inside_box(center_x, center_y, PASSPORT_NUMBER_TEXT_BOX):
                    number_box_items.append((center_y, center_x, str(text)))
        number_box_items.sort(key=lambda item: (item[0], item[1]))
        page_number_text = " ".join(item[2] for item in number_box_items)
        page_number_digits = re.sub(r"\D", "", page_number_text)
        if len(page_number_digits) == 6:
            number_digits = page_number_digits

    fields["passport_series_number_raw"] = " ".join(
        text for text in (series_text, number_text) if text
    )
    fields["passport_series"] = series_digits if len(series_digits) == 4 else ""
    fields["passport_number"] = (
        number_digits if len(number_digits) == 6 else ""
    )
    return fields


def process_image(image: np.ndarray, ocr: RapidOCR) -> dict:
    """Ищем и выравниваем страницу, затем распознаём её поля."""
    detection = find_page(image)
    if not detection["found"]:
        return {"found": False, "reason": detection["reason"]}

    page = straighten_candidate(
        image, np.asarray(detection["corners"], dtype=np.float32)
    )
    return {
        "found": True,
        "page_score": detection["score"],
        "fields": extract_fields(page, ocr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a synthetic passport page, straighten it, and read page 3 fields."
    )
    parser.add_argument("image", type=Path, help="Path to one input image.")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot read image: {args.image}")

    result = process_image(image, create_ocr())
    result["input"] = str(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
