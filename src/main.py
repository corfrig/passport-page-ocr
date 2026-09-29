import hashlib

import cv2
import numpy as np
import streamlit as st

if __package__:
    from .extract_passport_fields import FIELD_BOXES, PASSPORT_NUMBER_BOX, create_ocr, extract_fields
    from .passport_page_detector import find_page, straighten_candidate
else:
    from extract_passport_fields import FIELD_BOXES, PASSPORT_NUMBER_BOX, create_ocr, extract_fields
    from passport_page_detector import find_page, straighten_candidate


st.set_page_config(
    page_title="\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 OCR \u043f\u0430\u0441\u043f\u043e\u0440\u0442\u0430",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_ocr():
    return create_ocr()


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_page_outline(image: np.ndarray, corners: list) -> np.ndarray:
    preview = image.copy()
    points = np.asarray(corners, dtype=np.int32)
    cv2.polylines(preview, [points], True, (32, 160, 85), 6, cv2.LINE_AA)
    for point in points:
        cv2.circle(preview, tuple(point), 9, (32, 160, 85), -1, cv2.LINE_AA)
    return preview


def draw_field_boxes(page: np.ndarray) -> np.ndarray:
    preview = page.copy()
    regions = list(FIELD_BOXES.items()) + [
        ("passport_series_number", PASSPORT_NUMBER_BOX)
    ]
    colors = [
        (0, 130, 210),
        (155, 90, 0),
        (140, 40, 150),
        (0, 125, 50),
        (30, 70, 200),
        (170, 90, 20),
        (0, 0, 180),
    ]
    height, width = preview.shape[:2]
    thickness = max(2, round(min(height, width) / 500))

    for index, (name, box) in enumerate(regions):
        left, top, right, bottom = box
        x1, x2 = round(left * width), round(right * width)
        y1, y2 = round(top * height), round(bottom * height)
        color = colors[index % len(colors)]
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.putText(
            preview,
            name,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return preview


def render_saved_result(result: dict) -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("#### " + "1. \u0418\u0441\u0445\u043e\u0434\u043d\u043e\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435")
        source_preview = result.get("source_preview")
        if source_preview is not None:
            st.image(to_rgb(source_preview), width="stretch")
        else:
            st.info("\u0418\u0441\u0445\u043e\u0434\u043d\u043e\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e.")

    with right:
        st.markdown("#### " + "2. \u0421\u0442\u0440\u0430\u043d\u0438\u0446\u0430 \u0441 \u0431\u043e\u043a\u0441\u0430\u043c\u0438 \u043f\u043e\u043b\u0435\u0439")
        page_preview = result.get("page_preview")
        if page_preview is not None:
            st.image(to_rgb(page_preview), width="stretch")
        elif result.get("found") is False:
            st.info("\u0421\u0442\u0440\u0430\u043d\u0438\u0446\u0430 \u043f\u0430\u0441\u043f\u043e\u0440\u0442\u0430 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430.")
        else:
            st.info("\u0412\u044b\u043f\u0440\u044f\u043c\u043b\u0435\u043d\u043d\u0430\u044f \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430 \u043f\u043e\u043a\u0430 \u043d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u0430.")

    st.markdown("#### " + "3. \u0414\u0430\u043d\u043d\u044b\u0435 \u0432 JSON")
    st.json(result["json"], expanded=True)


def main() -> None:
    st.title("\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u0440\u0430\u0441\u043f\u043e\u0437\u043d\u0430\u0432\u0430\u043d\u0438\u044f \u043f\u0430\u0441\u043f\u043e\u0440\u0442\u0430")
    st.caption(
        "\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u0435 \u0441\u0438\u043d\u0442\u0435\u0442\u0438\u0447\u0435\u0441\u043a\u043e\u0435 \u0438\u043b\u0438 \u0440\u0430\u0437\u0440\u0435\u0448\u0451\u043d\u043d\u043e\u0435 \u0442\u0435\u0441\u0442\u043e\u0432\u043e\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435. \u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u043f\u043e\u043a\u0430\u0436\u0443 \u043d\u0430\u0439\u0434\u0435\u043d\u043d\u0443\u044e \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0443, \u0437\u0430\u0442\u0435\u043c \u0431\u043e\u043a\u0441\u044b \u043f\u043e\u043b\u0435\u0439 \u0438 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 OCR."
    )

    uploaded = st.file_uploader(
        "\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 \u043f\u0430\u0441\u043f\u043e\u0440\u0442\u0430",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="\u0424\u0430\u0439\u043b \u043e\u0431\u0440\u0430\u0431\u0430\u0442\u044b\u0432\u0430\u0435\u0442\u0441\u044f \u043b\u043e\u043a\u0430\u043b\u044c\u043d\u043e \u0432 \u044d\u0442\u043e\u043c \u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0438.",
    )
    file_bytes = uploaded.getvalue() if uploaded is not None else None
    upload_key = (
        hashlib.sha256(file_bytes).hexdigest() if file_bytes is not None else None
    )

    if st.session_state.get("upload_key") != upload_key:
        st.session_state["upload_key"] = upload_key
        st.session_state.pop("ocr_result", None)

    if uploaded is None:
        st.info("\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435, \u0447\u0442\u043e\u0431\u044b \u043d\u0430\u0447\u0430\u0442\u044c.")
        return

    if st.button("\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u0442\u044c \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435", type="primary"):
        with st.status("\u042d\u0442\u0430\u043f 1/3: \u0438\u0449\u0443 \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0443 \u043f\u0430\u0441\u043f\u043e\u0440\u0442\u0430\u2026", expanded=True) as status:
            try:
                if file_bytes is None:
                    raise ValueError("\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0442\u044c \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043d\u043d\u044b\u0439 \u0444\u0430\u0439\u043b.")

                encoded = np.frombuffer(file_bytes, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043e\u0442\u043a\u0440\u044b\u0442\u044c \u0444\u0430\u0439\u043b \u043a\u0430\u043a \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435.")

                detection = find_page(image)
                if not detection["found"]:
                    result = {
                        "input": uploaded.name,
                        "found": False,
                        "reason": detection["reason"],
                        "source_preview": image,
                        "page_preview": None,
                        "json": {
                            "input": uploaded.name,
                            "found": False,
                            "reason": detection["reason"],
                        },
                    }
                    status.update(
                        label="\u0421\u0442\u0440\u0430\u043d\u0438\u0446\u0430 \u043f\u0430\u0441\u043f\u043e\u0440\u0442\u0430 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430.",
                        state="error",
                        expanded=False,
                    )
                else:
                    page = straighten_candidate(
                        image,
                        np.asarray(detection["corners"], dtype=np.float32),
                    )
                    source_preview = draw_page_outline(image, detection["corners"])
                    st.markdown("#### " + "1. \u041d\u0430 \u0438\u0441\u0445\u043e\u0434\u043d\u043e\u043c \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0438 \u043d\u0430\u0439\u0434\u0435\u043d\u0430 \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430")
                    st.image(to_rgb(source_preview), width="stretch")

                    st.markdown("#### " + "2. \u0412\u044b\u043f\u0440\u044f\u043c\u043b\u0435\u043d\u043d\u0430\u044f \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430")
                    page_placeholder = st.empty()
                    page_placeholder.image(to_rgb(page), width="stretch")

                    status.update(
                        label="\u042d\u0442\u0430\u043f 2/3: \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430 \u043d\u0430\u0439\u0434\u0435\u043d\u0430. \u0420\u0430\u0441\u043f\u043e\u0437\u043d\u0430\u044e \u043f\u043e\u043b\u044f\u2026",
                        state="running",
                        expanded=True,
                    )
                    status.write(
                        "\u041f\u0440\u0438 \u043f\u0435\u0440\u0432\u043e\u043c \u0437\u0430\u043f\u0443\u0441\u043a\u0435 \u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u043c\u043e\u0436\u0435\u0442 \u0437\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c ONNX-\u043c\u043e\u0434\u0435\u043b\u044c."
                    )

                    fields = extract_fields(page, get_ocr())
                    page_preview = draw_field_boxes(page)
                    page_placeholder.image(to_rgb(page_preview), width="stretch")

                    result = {
                        "input": uploaded.name,
                        "found": True,
                        "page_score": detection["score"],
                        "fields": fields,
                        "source_preview": source_preview,
                        "page_preview": page_preview,
                        "json": {
                            "input": uploaded.name,
                            "found": True,
                            "page_score": detection["score"],
                            "fields": fields,
                        },
                    }
                    status.update(
                        label="\u042d\u0442\u0430\u043f 3/3: \u0433\u043e\u0442\u043e\u0432\u043e. \u0411\u043e\u043a\u0441\u044b \u043f\u043e\u043b\u0435\u0439 \u043f\u043e\u043a\u0430\u0437\u0430\u043d\u044b \u043d\u0430 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0438.",
                        state="complete",
                        expanded=False,
                    )

                st.session_state["ocr_result"] = result

            except Exception as error:
                error_result = {
                    "input": uploaded.name,
                    "found": False,
                    "error": str(error),
                }
                st.session_state["ocr_result"] = {
                    "json": error_result,
                    "source_preview": None,
                    "page_preview": None,
                }
                status.update(
                    label="\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430 \u0437\u0430\u0432\u0435\u0440\u0448\u0438\u043b\u0430\u0441\u044c \u0441 \u043e\u0448\u0438\u0431\u043a\u043e\u0439.",
                    state="error",
                    expanded=True,
                )
                st.error(f"{error}. " + "\u041f\u0440\u043e\u0432\u0435\u0440\u044c\u0442\u0435 \u0444\u0430\u0439\u043b \u0438 \u043f\u043e\u043f\u0440\u043e\u0431\u0443\u0439\u0442\u0435 \u0435\u0449\u0451 \u0440\u0430\u0437.")

    saved_result = st.session_state.get("ocr_result")
    if saved_result is not None:
        render_saved_result(saved_result)


if __name__ == "__main__":
    main()
