"""Compose special-cause matrix PNGs from exported matrix-cell images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable
import zipfile

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

CELL_METADATA_KEY = "special_cause_metadata"
MATRIX_METADATA_KEY = "special_cause_matrix_metadata"
MANIFEST_NAME = "special_cause_reciprocal_matrix_cells.json"
CELL_KIND = "special-cause-matrix-cell"
BUNDLE_KIND = "special-cause-matrix-cell-bundle"


@dataclass(frozen=True)
class CellImage:
    """One cropped matrix-cell image plus the metadata needed for placement."""

    metadata: dict[str, Any]
    image: Image.Image


class _BundleReader:
    def names(self) -> list[str]:
        raise NotImplementedError

    def read_bytes(self, name: str) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _DirectoryReader(_BundleReader):
    def __init__(self, path: Path):
        self.path = path
        self._names = [item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()]

    def names(self) -> list[str]:
        return list(self._names)

    def read_bytes(self, name: str) -> bytes:
        return (self.path / name).read_bytes()


class _ZipReader(_BundleReader):
    def __init__(self, path: Path):
        self.archive = zipfile.ZipFile(path)
        self._names = [name for name in self.archive.namelist() if not name.endswith("/")]

    def names(self) -> list[str]:
        return list(self._names)

    def read_bytes(self, name: str) -> bytes:
        return self.archive.read(name)

    def close(self) -> None:
        self.archive.close()


def _open_reader(path: Path) -> _BundleReader:
    if path.is_dir():
        return _DirectoryReader(path)
    if path.is_file() and zipfile.is_zipfile(path):
        return _ZipReader(path)
    raise ValueError(f"Input must be an exported ZIP bundle or directory: {path}")


def _load_json_bytes(raw: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"Metadata is not UTF-8 JSON: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must be a JSON object: {source}")
    return value


def _png_metadata_and_image(raw: bytes) -> tuple[dict[str, Any] | None, Image.Image]:
    image = Image.open(BytesIO(raw))
    metadata_text = image.info.get(CELL_METADATA_KEY)
    if metadata_text is None:
        metadata_text = getattr(image, "text", {}).get(CELL_METADATA_KEY)
    metadata = None
    if isinstance(metadata_text, str) and metadata_text.strip():
        metadata = _load_json_bytes(metadata_text.encode("utf-8"), source="embedded PNG metadata")
    return metadata, image.convert("RGBA")


def _find_manifest_name(names: Iterable[str]) -> str | None:
    for name in names:
        if Path(name).name == MANIFEST_NAME:
            return name
    return None


def _metadata_candidates(names: Iterable[str]) -> list[str]:
    return [name for name in names if name.lower().endswith(".json") and Path(name).name != MANIFEST_NAME]


def _png_candidates(names: Iterable[str]) -> list[str]:
    return [name for name in names if name.lower().endswith(".png")]


def _resolve_image_path(metadata: dict[str, Any], names: set[str]) -> str | None:
    image_path = metadata.get("image_path")
    if isinstance(image_path, str) and image_path in names:
        return image_path

    metadata_path = metadata.get("metadata_path")
    if isinstance(metadata_path, str):
        stem = Path(metadata_path).stem
        for prefix in ("cells", ""):
            candidate = f"{prefix + '/' if prefix else ''}{stem}.png"
            if candidate in names:
                return candidate

    stem = Path(str(image_path or metadata_path or "")).stem
    if stem:
        for name in names:
            path = Path(name)
            if path.suffix.lower() == ".png" and path.stem == stem:
                return name
    return None


def load_cell_bundle(input_path: str | Path) -> tuple[dict[str, Any], list[CellImage]]:
    """Load an exported special-cause cell bundle.

    ``input_path`` may be the ZIP downloaded by the GUI or an extracted directory.
    Metadata is read from the bundle manifest, JSON sidecars, or embedded PNG iTXt
    chunks, in that order.
    """

    path = Path(input_path)
    reader = _open_reader(path)
    try:
        names = reader.names()
        name_set = set(names)
        manifest_name = _find_manifest_name(names)
        manifest: dict[str, Any] = {}
        cells: list[CellImage] = []

        if manifest_name is not None:
            manifest = _load_json_bytes(reader.read_bytes(manifest_name), source=manifest_name)
            raw_cells = manifest.get("cells", [])
            if not isinstance(raw_cells, list):
                raise ValueError(f"Manifest cells must be a list: {manifest_name}")
            for index, raw_metadata in enumerate(raw_cells):
                if not isinstance(raw_metadata, dict):
                    raise ValueError(f"Manifest cell {index} is not an object")
                metadata = dict(raw_metadata)
                metadata_path = metadata.get("metadata_path")
                if isinstance(metadata_path, str) and metadata_path in name_set:
                    sidecar_metadata = _load_json_bytes(reader.read_bytes(metadata_path), source=metadata_path)
                    metadata.update(sidecar_metadata)
                image_path = _resolve_image_path(metadata, name_set)
                if image_path is None:
                    raise ValueError(f"Could not find PNG for cell metadata entry {index}")
                png_metadata, image = _png_metadata_and_image(reader.read_bytes(image_path))
                if png_metadata:
                    metadata.update(png_metadata)
                metadata.setdefault("image_path", image_path)
                cells.append(CellImage(metadata=metadata, image=image))
        else:
            for metadata_name in _metadata_candidates(names):
                metadata = _load_json_bytes(reader.read_bytes(metadata_name), source=metadata_name)
                if metadata.get("kind") != CELL_KIND:
                    continue
                metadata.setdefault("metadata_path", metadata_name)
                image_path = _resolve_image_path(metadata, name_set)
                if image_path is None:
                    continue
                png_metadata, image = _png_metadata_and_image(reader.read_bytes(image_path))
                if png_metadata:
                    metadata.update(png_metadata)
                metadata.setdefault("image_path", image_path)
                cells.append(CellImage(metadata=metadata, image=image))

            if not cells:
                for image_name in _png_candidates(names):
                    metadata, image = _png_metadata_and_image(reader.read_bytes(image_name))
                    if metadata and metadata.get("kind") == CELL_KIND:
                        metadata.setdefault("image_path", image_name)
                        cells.append(CellImage(metadata=metadata, image=image))

        if not cells:
            raise ValueError("No special-cause matrix cell images were found.")
        return manifest, cells
    finally:
        reader.close()


def _numeric_key(value: Any, *, ndigits: int = 6) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return round(numeric, ndigits)


def _cell_key(L: Any, theta_deg: Any) -> tuple[int, float]:
    return int(round(float(L))), _numeric_key(theta_deg)


def _ordered_values_from_manifest_or_cells(
    manifest: dict[str, Any],
    cells: list[CellImage],
    key: str,
    metadata_key: str,
) -> list[float | int]:
    manifest_values = manifest.get(key)
    if isinstance(manifest_values, list) and manifest_values:
        return [int(value) if key == "L_values" else float(value) for value in manifest_values]

    values = []
    seen = set()
    for cell in cells:
        value = cell.metadata.get(metadata_key)
        if value is None:
            continue
        normalized = int(round(float(value))) if key == "L_values" else _numeric_key(value)
        if normalized not in seen:
            seen.add(normalized)
            values.append(normalized)
    return sorted(values)


def _bbox(metadata: dict[str, Any], image: Image.Image) -> dict[str, float]:
    raw = metadata.get("bragg_bbox_in_crop_px")
    if isinstance(raw, dict):
        try:
            return {
                "x": float(raw["x"]),
                "y": float(raw["y"]),
                "width": float(raw["width"]),
                "height": float(raw["height"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    return {"x": 0.0, "y": 0.0, "width": float(image.width), "height": float(image.height)}


def _layout_defaults(width: int, height: int, colorbar: bool) -> dict[str, int]:
    return {
        "outer_margin": 46,
        "row_label_band": 78,
        "colorbar_band": 170 if colorbar else 28,
        "title_band": 58,
        "column_label_band": 54,
        "bottom_margin": 46,
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_centered_text(
    image: Image.Image,
    text: str,
    x: float,
    y: float,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int] = (45, 67, 99, 255),
    angle: float = 0.0,
) -> None:
    draw = ImageDraw.Draw(image)
    if not angle:
        width, height = _text_size(draw, text, font)
        draw.text((x - width / 2, y - height / 2), text, font=font, fill=fill)
        return

    width, height = _text_size(draw, text, font)
    label = Image.new("RGBA", (max(1, width + 8), max(1, height + 8)), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((4, 4), text, font=font, fill=fill)
    rotated = label.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(rotated, (int(round(x - rotated.width / 2)), int(round(y - rotated.height / 2))))


def _draw_colorbar(
    image: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    draw = ImageDraw.Draw(image)
    for offset in range(height):
        t = 1.0 - offset / max(1, height - 1)
        red = int(round(128 + (255 - 128) * t))
        green = int(round(128 * (1.0 - t)))
        blue = int(round(128 * (1.0 - t)))
        draw.line((x, y + offset, x + width - 1, y + offset), fill=(red, green, blue, 255))

    label_font = _font(16)
    tick_font = _font(14)
    _draw_centered_text(image, "Mosaic", x + width / 2, y - 32, font=label_font)
    _draw_centered_text(image, "Intensity", x + width / 2, y - 14, font=label_font)
    for tick in (1, 0.8, 0.6, 0.4, 0.2, 0):
        tick_y = y + (1 - tick) * height
        draw.text((x + width + 8, tick_y - 7), str(tick), font=tick_font, fill=(45, 67, 99, 255))


def _paste_clipped(
    canvas: Image.Image,
    sprite: Image.Image,
    x: float,
    y: float,
    cell_rect: tuple[float, float, float, float],
    *,
    clip: bool,
) -> None:
    paste_x = int(round(x))
    paste_y = int(round(y))
    if not clip:
        canvas.alpha_composite(sprite, (paste_x, paste_y))
        return

    cell_x, cell_y, cell_width, cell_height = cell_rect
    left = max(paste_x, int(math.floor(cell_x)), 0)
    top = max(paste_y, int(math.floor(cell_y)), 0)
    right = min(paste_x + sprite.width, int(math.ceil(cell_x + cell_width)), canvas.width)
    bottom = min(paste_y + sprite.height, int(math.ceil(cell_y + cell_height)), canvas.height)
    if right <= left or bottom <= top:
        return
    crop = sprite.crop((left - paste_x, top - paste_y, right - paste_x, bottom - paste_y))
    canvas.alpha_composite(crop, (left, top))


def _file_number_token(value: float | int, *, width: int = 3) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value)).zfill(width)
    whole, fraction = f"{numeric_value:g}".split(".", 1)
    return f"{whole.zfill(width)}p{fraction}"


def _scale_override_key(metadata: dict[str, Any]) -> str:
    L = int(round(float(metadata["L"])))
    theta = _numeric_key(metadata["theta_deg"])
    return f"L{L:03d}_theta{_file_number_token(theta)}"


def _parse_scale_override(raw: str) -> tuple[tuple[int, float] | str, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Scale overrides must use KEY=SCALE.")
    key_text, scale_text = raw.split("=", 1)
    try:
        scale = float(scale_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid scale value: {scale_text}") from exc
    if scale <= 0 or not math.isfinite(scale):
        raise argparse.ArgumentTypeError("Scale must be a positive finite number.")

    compact = key_text.strip().lower().replace(" ", "")
    if "," in compact:
        left, right = compact.split(",", 1)
        try:
            return _cell_key(float(left.lstrip("l=")), float(right.lstrip("theta="))), scale
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid L,theta scale key: {key_text}") from exc

    match = re.search(r"l0*(\d+).*theta0*([0-9]+(?:p[0-9]+)?)", compact)
    if match:
        theta = float(match.group(2).replace("p", "."))
        return _cell_key(int(match.group(1)), theta), scale
    return key_text.strip(), scale


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("Value must be a positive finite number.")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return value


def compose_matrix_image(
    manifest: dict[str, Any],
    cells: list[CellImage],
    *,
    width: int = 1800,
    height: int | None = None,
    bragg_fill_fraction: float | None = None,
    cell_scale: float = 1.0,
    scale_overrides: dict[tuple[int, float] | str, float] | None = None,
    preserve_relative_l_scale: bool | None = None,
    title: str = "Special Cause Reciprocal Matrix",
    colorbar: bool = True,
    clip_cells: bool = True,
    transparent_background: bool = False,
    debug_boxes: bool = False,
    layout: dict[str, int] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Return a composed 3x3 matrix image and placement metadata."""

    if height is None:
        height = width
    l_values = _ordered_values_from_manifest_or_cells(manifest, cells, "L_values", "L")
    theta_values = _ordered_values_from_manifest_or_cells(manifest, cells, "theta_values", "theta_deg")
    if len(l_values) != 3 or len(theta_values) != 3:
        raise ValueError(f"Expected a 3x3 bundle, found L={l_values} and theta={theta_values}.")

    cells_by_key = {_cell_key(cell.metadata["L"], cell.metadata["theta_deg"]): cell for cell in cells}
    defaults = manifest if manifest else (cells[0].metadata.get("matrix_defaults") or {})
    if bragg_fill_fraction is None:
        bragg_fill_fraction = float(defaults.get("bragg_cell_fill_fraction", 0.82))
    if preserve_relative_l_scale is None:
        preserve_relative_l_scale = bool(defaults.get("preserve_relative_l_scale", False))
    scale_overrides = scale_overrides or {}
    layout_values = _layout_defaults(width, height, colorbar)
    if layout:
        layout_values.update(layout)

    background = (0, 0, 0, 0) if transparent_background else (255, 255, 255, 255)
    canvas = Image.new("RGBA", (width, height), background)
    draw = ImageDraw.Draw(canvas)

    outer_margin = layout_values["outer_margin"]
    row_label_band = layout_values["row_label_band"]
    colorbar_band = layout_values["colorbar_band"]
    title_band = layout_values["title_band"]
    column_label_band = layout_values["column_label_band"]
    bottom_margin = layout_values["bottom_margin"]
    grid_x = outer_margin + row_label_band
    grid_y = outer_margin + title_band + column_label_band
    grid_width = width - grid_x - colorbar_band - outer_margin
    grid_height = height - grid_y - bottom_margin
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("Layout margins leave no room for the matrix grid.")
    cell_width = grid_width / len(theta_values)
    cell_height = grid_height / len(l_values)

    _draw_centered_text(canvas, title, width / 2, outer_margin * 0.75, font=_font(24))
    for col_index, theta in enumerate(theta_values):
        label = f"θᵢ = {theta:g}°"
        _draw_centered_text(
            canvas,
            label,
            grid_x + cell_width * (col_index + 0.5),
            outer_margin + title_band + column_label_band * 0.45,
            font=_font(20),
        )
    for row_index, L in enumerate(l_values):
        _draw_centered_text(
            canvas,
            f"L = {int(L)}",
            outer_margin + row_label_band * 0.32,
            grid_y + cell_height * (row_index + 0.5),
            font=_font(20),
            angle=90,
        )

    placements = []
    for row_index, L in enumerate(l_values):
        for col_index, theta in enumerate(theta_values):
            key = _cell_key(L, theta)
            cell = cells_by_key.get(key)
            if cell is None:
                raise ValueError(f"Missing matrix cell L={L}, theta={theta}.")
            image = cell.image.convert("RGBA")
            bragg = _bbox(cell.metadata, image)
            relative_extent = float(cell.metadata.get("relative_extent", 1.0) or 1.0)
            relative_scale = relative_extent if preserve_relative_l_scale else 1.0
            bragg_extent = max(1.0, float(bragg["width"]), float(bragg["height"]))
            override_key = _scale_override_key(cell.metadata)
            override = scale_overrides.get(key, scale_overrides.get(override_key, 1.0))
            scale = bragg_fill_fraction * min(cell_width, cell_height) * relative_scale / bragg_extent
            scale *= cell_scale * override
            draw_width = max(1, int(round(image.width * scale)))
            draw_height = max(1, int(round(image.height * scale)))
            resized = image.resize((draw_width, draw_height), Image.Resampling.LANCZOS)

            cell_x = grid_x + col_index * cell_width
            cell_y = grid_y + row_index * cell_height
            bragg_center_x = (float(bragg["x"]) + float(bragg["width"]) / 2.0) * scale
            bragg_center_y = (float(bragg["y"]) + float(bragg["height"]) / 2.0) * scale
            cell_center_x = cell_x + cell_width / 2.0
            cell_center_y = cell_y + cell_height / 2.0
            paste_x = cell_center_x - bragg_center_x
            paste_y = cell_center_y - bragg_center_y
            _paste_clipped(
                canvas,
                resized,
                paste_x,
                paste_y,
                (cell_x, cell_y, cell_width, cell_height),
                clip=clip_cells,
            )

            placement = {
                "L": int(L),
                "theta_deg": float(theta),
                "image_path": cell.metadata.get("image_path"),
                "cell_rect_px": {"x": cell_x, "y": cell_y, "width": cell_width, "height": cell_height},
                "paste_rect_px": {"x": paste_x, "y": paste_y, "width": draw_width, "height": draw_height},
                "scale": scale,
                "scale_override": override,
                "bragg_fill_fraction": max(float(bragg["width"]), float(bragg["height"])) * scale / min(cell_width, cell_height),
            }
            placements.append(placement)

            if debug_boxes:
                draw.rectangle((cell_x, cell_y, cell_x + cell_width, cell_y + cell_height), outline=(35, 35, 35, 128), width=1)
                draw.rectangle((paste_x, paste_y, paste_x + draw_width, paste_y + draw_height), outline=(214, 60, 130, 180), width=2)
                bragg_x = paste_x + float(bragg["x"]) * scale
                bragg_y = paste_y + float(bragg["y"]) * scale
                draw.rectangle(
                    (
                        bragg_x,
                        bragg_y,
                        bragg_x + float(bragg["width"]) * scale,
                        bragg_y + float(bragg["height"]) * scale,
                    ),
                    outline=(42, 126, 255, 220),
                    width=2,
                )

    if colorbar:
        colorbar_x = int(round(width - outer_margin - colorbar_band * 0.42))
        colorbar_y = int(round(grid_y + grid_height * 0.16))
        colorbar_width = 30
        colorbar_height = int(round(grid_height * 0.68))
        _draw_colorbar(canvas, colorbar_x, colorbar_y, colorbar_width, colorbar_height)

    metadata = {
        "kind": "special-cause-matrix-composite",
        "version": 1,
        "output_size_px": {"width": width, "height": height},
        "L_values": [int(value) for value in l_values],
        "theta_values": [float(value) for value in theta_values],
        "bragg_fill_fraction": bragg_fill_fraction,
        "cell_scale": cell_scale,
        "preserve_relative_l_scale": preserve_relative_l_scale,
        "layout": layout_values,
        "colorbar": colorbar,
        "clip_cells": clip_cells,
        "placements": placements,
    }
    return canvas, metadata


def save_matrix_image(image: Image.Image, output_path: str | Path, metadata: dict[str, Any]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text(MATRIX_METADATA_KEY, json.dumps(metadata, indent=2, sort_keys=True))
    image.save(output, pnginfo=pnginfo)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 3x3 special-cause matrix PNG from exported cropped cell images.",
    )
    parser.add_argument("input", help="Exported special_cause_reciprocal_matrix_cells.zip or extracted directory.")
    parser.add_argument("--output", "-o", default="special_cause_reciprocal_matrix.png", help="Output PNG path.")
    parser.add_argument("--width", type=_positive_int, default=1800, help="Output width in pixels.")
    parser.add_argument("--height", type=_positive_int, default=None, help="Output height in pixels. Defaults to width.")
    parser.add_argument("--bragg-fill", type=_positive_float, default=None, help="Target Bragg footprint fraction in each cell.")
    parser.add_argument("--cell-scale", type=_positive_float, default=1.0, help="Global multiplier for all cropped cell images.")
    parser.add_argument(
        "--scale",
        dest="scale_overrides",
        action="append",
        type=_parse_scale_override,
        default=[],
        help="Per-cell multiplier. Use L003_theta005=1.10 or 3,5=1.10. May be repeated.",
    )
    parser.add_argument("--relative-l-scale", action="store_true", help="Preserve relative L size across rows.")
    parser.add_argument("--local-scale", action="store_true", help="Scale each row locally so its Bragg sphere fills the same cell fraction.")
    parser.add_argument("--no-colorbar", action="store_true", help="Do not draw the shared mosaic intensity colorbar.")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip cell images to their grid cells.")
    parser.add_argument("--transparent-background", action="store_true", help="Use a transparent background instead of white.")
    parser.add_argument("--debug-boxes", action="store_true", help="Draw cell, image, and Bragg bounding boxes.")
    parser.add_argument("--metadata-output", default=None, help="Optional JSON sidecar path for composite metadata.")
    parser.add_argument("--outer-margin", type=int, default=None)
    parser.add_argument("--row-label-band", type=int, default=None)
    parser.add_argument("--colorbar-band", type=int, default=None)
    parser.add_argument("--title-band", type=int, default=None)
    parser.add_argument("--column-label-band", type=int, default=None)
    parser.add_argument("--bottom-margin", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest, cells = load_cell_bundle(args.input)
    overrides = dict(args.scale_overrides)
    preserve_relative_l_scale = None
    if args.relative_l_scale and args.local_scale:
        raise SystemExit("Use only one of --relative-l-scale or --local-scale.")
    if args.relative_l_scale:
        preserve_relative_l_scale = True
    elif args.local_scale:
        preserve_relative_l_scale = False

    layout = {
        key: value
        for key, value in {
            "outer_margin": args.outer_margin,
            "row_label_band": args.row_label_band,
            "colorbar_band": args.colorbar_band,
            "title_band": args.title_band,
            "column_label_band": args.column_label_band,
            "bottom_margin": args.bottom_margin,
        }.items()
        if value is not None
    }
    image, metadata = compose_matrix_image(
        manifest,
        cells,
        width=args.width,
        height=args.height,
        bragg_fill_fraction=args.bragg_fill,
        cell_scale=args.cell_scale,
        scale_overrides=overrides,
        preserve_relative_l_scale=preserve_relative_l_scale,
        colorbar=not args.no_colorbar,
        clip_cells=not args.no_clip,
        transparent_background=args.transparent_background,
        debug_boxes=args.debug_boxes,
        layout=layout,
    )
    output = Path(args.output)
    save_matrix_image(image, output, metadata)
    if args.metadata_output:
        metadata_path = Path(args.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
