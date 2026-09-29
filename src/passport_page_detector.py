"""Поиск паспортной страницы по форме и красной защитной полосе."""

from collections.abc import Iterable, Iterator
from typing import NamedTuple

import cv2
import numpy as np


# Пороги подбирали под этот макет.
MAX_ANALYSIS_SIDE = 1400
FULL_FRAME_RED_BAND_THRESHOLD = 0.80
WHITE_BACKGROUND_THRESHOLD = 240
WHITE_BACKGROUND_FRACTION = 0.70
MIN_CANDIDATE_RED_BAND = 0.22
CANDIDATE_AREA_BONUS = 0.12


class Candidate(NamedTuple):
    """Углы страницы-кандидата и занимаемая ею часть изображения."""

    corners: np.ndarray
    area_fraction: float


class RankedCandidate(NamedTuple):
    """Кандидат и оценки, по которым выбираем лучший вариант."""

    rank: float
    corners: np.ndarray
    area_fraction: float
    red_band_score: float


def order_corners(points: np.ndarray) -> np.ndarray:
    """Расставляем углы по часовой стрелке, начиная с верхнего левого."""
    points = points.reshape(4, 2).astype(np.float32)

    coordinate_sums = points.sum(axis=1)

    vertical_minus_horizontal = np.diff(points, axis=1).ravel()

    return np.array([
        points[np.argmin(coordinate_sums)],
        points[np.argmin(vertical_minus_horizontal)],
        points[np.argmax(coordinate_sums)],
        points[np.argmax(vertical_minus_horizontal)],
    ], dtype=np.float32)


def straighten_candidate(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Выпрямляем кандидата, чтобы проверить, где у него красная полоса."""
    top_left, top_right, bottom_right, bottom_left = order_corners(corners)

    width = int(max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    ))
    height = int(max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    ))
    width, height = max(1, width), max(1, height)

    rectangle_corners = np.array([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    page_corners = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
    )

    perspective = cv2.getPerspectiveTransform(page_corners, rectangle_corners)
    page = cv2.warpPerspective(image, perspective, (width, height))
    quarter_turns = red_band_orientation(page)
    if quarter_turns:
        page = np.rot90(page, quarter_turns)
    return np.ascontiguousarray(page)


def _red_band_scores(image: np.ndarray) -> list[float]:
    """Считаем, насколько каждый край похож на красную полосу паспорта."""
    small_image = cv2.resize(image, (320, 220), interpolation=cv2.INTER_AREA)

    blue, green, red = cv2.split(small_image.astype(np.float32))
    red_pixel_mask = (
        (red > 40)
        & (red > green * 1.12)
        & (red > blue * 1.10)
        & (blue > red * 0.43)
    )

    scores_by_orientation = []
    for quarter_turns in range(4):
        oriented_mask = np.rot90(red_pixel_mask, quarter_turns)
        edge_height = max(4, int(oriented_mask.shape[0] * 0.08))
        pixels_near_top_edge = oriented_mask[:edge_height]

        coverage_by_row = pixels_near_top_edge.mean(axis=1)
        strongest_rows = np.sort(coverage_by_row)[-5:]
        scores_by_orientation.append(float(strongest_rows.mean()))

    return scores_by_orientation


def red_band_orientation(image: np.ndarray) -> int:
    """Возвращаем поворот, при котором красная полоса окажется сверху."""
    return int(np.argmax(_red_band_scores(image)))


def red_band_strength(image: np.ndarray) -> float:
    """Возвращаем самую высокую оценку красной полосы на краю."""
    return max(_red_band_scores(image))


def scanner_candidates(
    image: np.ndarray, gray: np.ndarray
) -> Iterator[Candidate]:
    """Ищем страницу на почти белом фоне сканера."""
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]

    candidate_masks = [
        cv2.inRange(gray, 0, 238),
        cv2.inRange(saturation, 18, 255),
    ]
    close_kernel = np.ones((13, 13), np.uint8)

    for candidate_mask in candidate_masks:
        connected_regions = cv2.morphologyEx(
            candidate_mask, cv2.MORPH_CLOSE, close_kernel
        )
        contours, _ = cv2.findContours(
            connected_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        largest_contours = sorted(
            contours, key=cv2.contourArea, reverse=True
        )[:5]
        for contour in largest_contours:
            area_fraction = cv2.contourArea(contour) / gray.size
            if not 0.04 <= area_fraction <= 0.60:
                continue

            rectangle = cv2.minAreaRect(contour)
            corners = order_corners(cv2.boxPoints(rectangle))
            yield Candidate(corners, area_fraction)


def photo_candidates(gray: np.ndarray) -> Iterator[Candidate]:
    """Ищем на фотографии контуры, похожие на страницу."""
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_mask = cv2.Canny(blurred_image, 35, 110)
    connected_edges = cv2.morphologyEx(
        edge_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    contours, _ = cv2.findContours(
        connected_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    image_height, image_width = gray.shape

    largest_contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )[:120]
    for contour in largest_contours:
        area_fraction = cv2.contourArea(contour) / gray.size
        if not 0.025 <= area_fraction <= 0.985:
            continue

        perimeter = cv2.arcLength(contour, True)
        for approximation_tolerance in (0.018, 0.035, 0.06):
            polygon = cv2.approxPolyDP(
                contour, approximation_tolerance * perimeter, True
            )
            if len(polygon) != 4:
                continue
            if not cv2.isContourConvex(polygon):
                continue

            corners = order_corners(polygon)
            spans_full_width = (
                np.any(corners[:, 0] < 2)
                and np.any(corners[:, 0] > image_width - 3)
            )
            spans_full_height = (
                np.any(corners[:, 1] < 2)
                and np.any(corners[:, 1] > image_height - 3)
            )
            if spans_full_width or spans_full_height:
                continue

            side_lengths = np.linalg.norm(
                np.roll(corners, -1, axis=0) - corners, axis=1
            )
            if min(side_lengths) < 0.07 * min(image_width, image_height):
                continue

            longest_to_shortest_side = max(side_lengths) / min(side_lengths)
            if not 1.05 <= longest_to_shortest_side <= 2.8:
                continue

            yield Candidate(corners, area_fraction)
            break


def best_candidate(
    image: np.ndarray, candidates: Iterable[Candidate]
) -> RankedCandidate | None:
    """Выбираем кандидата с самой заметной красной полосой."""
    matches = []
    for candidate in candidates:
        straight_image = straighten_candidate(image, candidate.corners)
        red_band_score = red_band_strength(straight_image)
        if red_band_score < MIN_CANDIDATE_RED_BAND:
            continue

        rank = red_band_score + CANDIDATE_AREA_BONUS * np.sqrt(
            candidate.area_fraction
        )
        matches.append(RankedCandidate(
            rank=rank,
            corners=candidate.corners,
            area_fraction=candidate.area_fraction,
            red_band_score=red_band_score,
        ))

    return max(matches, key=lambda match: match.rank) if matches else None


def find_page(image: np.ndarray | None) -> dict:
    """Ищем страницу и возвращаем её углы с оценками или причину отказа."""
    if image is None or image.size == 0:
        return {"found": False, "reason": "image_unreadable"}

    original_height, original_width = image.shape[:2]
    scale = min(
        1.0,
        MAX_ANALYSIS_SIDE / max(original_width, original_height),
    )
    analysis_image = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    analysis_height, analysis_width = analysis_image.shape[:2]
    gray = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)

    full_frame_band_score = red_band_strength(analysis_image)
    if full_frame_band_score >= FULL_FRAME_RED_BAND_THRESHOLD:
        full_frame_corners = np.array([
            [0, 0],
            [analysis_width - 1, 0],
            [analysis_width - 1, analysis_height - 1],
            [0, analysis_height - 1],
        ], dtype=np.float32)
        match = RankedCandidate(
            rank=full_frame_band_score,
            corners=full_frame_corners,
            area_fraction=1.0,
            red_band_score=full_frame_band_score,
        )
    else:
        match = None

        white_pixel_fraction = float(
            (gray > WHITE_BACKGROUND_THRESHOLD).mean()
        )
        if white_pixel_fraction > WHITE_BACKGROUND_FRACTION:
            match = best_candidate(
                analysis_image,
                scanner_candidates(analysis_image, gray),
            )

        if match is None:
            match = best_candidate(
                analysis_image,
                photo_candidates(gray),
            )

    if match is None:
        return {"found": False, "reason": "no_plausible_page"}

    original_corners = np.rint(match.corners / scale).astype(int).tolist()
    return {
        "found": True,
        "corners": original_corners,
        "score": round(float(match.rank), 3),
        "area_fraction": round(float(match.area_fraction), 3),
        "red_strip_score": round(float(match.red_band_score), 3),
    }


def draw_result(image: np.ndarray, result: dict) -> np.ndarray:
    """Рисуем рамку найденной страницы или причину отказа."""
    annotated_image = image.copy()
    line_width = max(3, round(min(annotated_image.shape[:2]) / 300))

    if result["found"]:
        corners = np.asarray(result["corners"], dtype=np.int32)
        cv2.polylines(
            annotated_image, [corners], True, (25, 220, 25), line_width, cv2.LINE_AA
        )
        for point in corners:
            cv2.circle(
                annotated_image, tuple(point), line_width * 2,
                (10, 80, 255), -1, cv2.LINE_AA,
            )
        label = f"PAGE  score={result['score']:.2f}"
        label_color = (25, 170, 25)
    else:
        label = f"NO PAGE  {result['reason']}"
        label_color = (20, 30, 220)

    cv2.putText(
        annotated_image,
        label,
        (20, max(40, line_width * 9)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, line_width / 4),
        label_color,
        max(2, line_width // 2),
        cv2.LINE_AA,
    )
    return annotated_image
