from io import BytesIO
import json
import zipfile

from PIL import Image, ImageDraw, PngImagePlugin

from special_cause_matrix_cells import (
    CELL_METADATA_KEY,
    MATRIX_METADATA_KEY,
    compose_matrix_image,
    load_cell_bundle,
    save_matrix_image,
)


def _png_bytes(metadata):
    image = Image.new("RGBA", (120, 90), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 10, 85, 70), fill=(180, 180, 180, 255))
    draw.arc((40, 22, 100, 82), start=20, end=220, fill=(255, 0, 0, 255), width=4)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text(CELL_METADATA_KEY, json.dumps(metadata))
    buffer = BytesIO()
    image.save(buffer, format="PNG", pnginfo=pnginfo)
    return buffer.getvalue()


def _write_dummy_bundle(path):
    cells = []
    with zipfile.ZipFile(path, "w") as archive:
        for row, L in enumerate((3, 6, 9), start=1):
            for col, theta in enumerate((5.0, 10.0, 15.0), start=1):
                stem = f"special_cause_L{L:03d}_theta{int(theta):03d}"
                image_path = f"cells/{stem}.png"
                metadata_path = f"metadata/{stem}.json"
                metadata = {
                    "kind": "special-cause-matrix-cell",
                    "version": 1,
                    "image_path": image_path,
                    "metadata_path": metadata_path,
                    "row": row,
                    "col": col,
                    "L": L,
                    "theta_deg": theta,
                    "relative_extent": L / 9,
                    "crop_px": {"width": 120, "height": 90},
                    "bragg_bbox_in_crop_px": {"x": 25, "y": 10, "width": 60, "height": 60},
                    "trace_names": ["Bragg sphere", "Bragg/Ewald overlap", "Ewald sphere"],
                    "matrix_defaults": {
                        "theta_values": [5.0, 10.0, 15.0],
                        "L_values": [3, 6, 9],
                        "bragg_cell_fill_fraction": 0.82,
                        "preserve_relative_l_scale": False,
                    },
                }
                archive.writestr(image_path, _png_bytes(metadata))
                archive.writestr(metadata_path, json.dumps(metadata))
                cells.append(metadata)
        archive.writestr(
            "special_cause_reciprocal_matrix_cells.json",
            json.dumps(
                {
                    "kind": "special-cause-matrix-cell-bundle",
                    "version": 1,
                    "theta_values": [5.0, 10.0, 15.0],
                    "L_values": [3, 6, 9],
                    "bragg_cell_fill_fraction": 0.82,
                    "preserve_relative_l_scale": False,
                    "cells": cells,
                }
            ),
        )


def test_special_cause_matrix_cell_bundle_composes_resizable_matrix(tmp_path):
    bundle = tmp_path / "special_cause_reciprocal_matrix_cells.zip"
    _write_dummy_bundle(bundle)

    manifest, cells = load_cell_bundle(bundle)
    image, metadata = compose_matrix_image(
        manifest,
        cells,
        width=900,
        bragg_fill_fraction=0.70,
        cell_scale=1.05,
        scale_overrides={(3, 5.0): 1.20},
        debug_boxes=True,
    )

    assert image.size == (900, 900)
    assert len(cells) == 9
    assert len(metadata["placements"]) == 9
    first = next(item for item in metadata["placements"] if item["L"] == 3 and item["theta_deg"] == 5.0)
    assert first["scale_override"] == 1.20
    assert metadata["bragg_fill_fraction"] == 0.70
    assert metadata["cell_scale"] == 1.05

    output = tmp_path / "matrix.png"
    save_matrix_image(image, output, metadata)
    saved = Image.open(output)
    assert saved.size == (900, 900)
    assert json.loads(saved.info[MATRIX_METADATA_KEY])["kind"] == "special-cause-matrix-composite"
