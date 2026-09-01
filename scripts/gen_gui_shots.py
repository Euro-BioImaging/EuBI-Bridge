"""Generate documentation screenshots of the EuBI-Bridge GUI.

Run it after changing the interface so the documentation images cannot drift
away from what the program actually shows::

    python docs/scripts/gen_gui_shots.py                 # all shots
    python docs/scripts/gen_gui_shots.py --only batch    # just the batch ones
    python docs/scripts/gen_gui_shots.py --list          # names, take nothing

Images are written to ``docs/images/gui/``.

Each shot stages realistic state first (a queued batch, a few overridden cells)
because an empty interface documents nothing.  Nothing is written outside the
output directory and no conversion is started, so the script is safe to run
repeatedly.

Rendering notes
---------------
Qt is driven offscreen, which needs no display but *does* need fonts: on a
machine without any installed, every label renders as a box.  The script checks
for this and refuses rather than writing images that look subtly fine in a
thumbnail and are useless at full size.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUTPUT_DIR = _ROOT / "docs" / "images" / "gui"

# Big enough that panels are not cramped, small enough to stay readable when a
# docs theme scales the image down.
WINDOW_SIZE = (1400, 900)
DIALOG_SIZE = (620, 560)

# Documentation images use one fixed appearance so they stay consistent with
# each other and across regenerations, whatever the author's own settings are.
THEME = "Dark (default)"
FONT_POINT_SIZE = 9


# ── environment ───────────────────────────────────────────────────────────────

#: Where each platform keeps its fonts.  The offscreen backend does not consult
#: the system font configuration, and PyQt6 no longer ships fonts of its own, so
#: it finds none unless pointed at a directory explicitly.
_SYSTEM_FONT_DIRS = {
    "win32": ["C:/Windows/Fonts"],
    "darwin": ["/System/Library/Fonts", "/Library/Fonts"],
}
_LINUX_FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts"]


def _configure_qt() -> None:
    """Offscreen rendering, deterministic scale, before Qt is imported."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # A stray scale factor from the user's settings would change every image.
    os.environ["QT_SCALE_FACTOR"] = "1"

    if "QT_QPA_FONTDIR" not in os.environ:
        candidates = _SYSTEM_FONT_DIRS.get(sys.platform, _LINUX_FONT_DIRS)
        for directory in candidates:
            if Path(directory).is_dir():
                os.environ["QT_QPA_FONTDIR"] = directory
                break


def _apply_appearance(app) -> None:
    """Pin the palette and font used for every image.

    ``settings_dialog.apply_settings`` is not used: it persists to the user's
    settings file, so generating documentation would silently change their
    theme.  The same palette and stylesheet are applied directly instead.
    """
    from PyQt6.QtGui import QFont
    from eubi_bridge.qt_gui.settings_dialog import (
        PALETTES, STYLESHEETS, _BASE_STYLESHEET)

    app.setPalette(PALETTES[THEME])
    app.setStyleSheet(STYLESHEETS.get(THEME, _BASE_STYLESHEET))

    font = QFont(app.font())
    font.setPointSize(FONT_POINT_SIZE)
    app.setFont(font)
    app.processEvents()


def _fonts_available(app) -> bool:
    """True when Qt can actually draw text.

    Offscreen Qt renders every glyph as a box when it can find no fonts, which
    happens even on a machine full of them because the backend ignores the
    system font configuration.  :func:`_configure_qt` points it at the usual
    directory first; this verifies that it worked, because the images still
    *look* plausible at thumbnail size and the problem would otherwise surface
    only in review.
    """
    from PyQt6.QtGui import QFontDatabase
    return bool(QFontDatabase.families())


# ── staging helpers ───────────────────────────────────────────────────────────

def _sample_batch(page) -> None:
    """Queue a few rows with per-row overrides, as a real batch would look."""
    from copy import deepcopy

    config = page._ui_to_config()
    page._batch.set_baseline(deepcopy(config))
    page._batch.add(
        config,
        ["/data/experiment/plate1/well_A1.czi",
         "/data/experiment/plate1/well_A2.czi",
         "/data/experiment/plate2/well_A1.czi"],
        "/data/converted",
    )
    # Overrides that exercise the interesting rendering: a differing value, an
    # inert cell, and the collision that parent-prefixing resolves.
    page._batch.update_cells([0], "dtype", "uint8")
    page._batch.update_cells([1], "auto_chunk", False)
    page._batch.update_cells([1], "z_chunk", 64)
    page._batch.update_cells([2], "n_layers", 3)
    page._refresh_batch_table()


def _select_cells(page, rows, columns) -> None:
    """Select a rectangle of cells so Edit Cells has something to act on."""
    table = page._batch_table
    table.clearSelection()
    for row in rows:
        for column in columns:
            item = table.item(row, column)
            if item is not None and item.flags():
                item.setSelected(True)


# ── the shots ─────────────────────────────────────────────────────────────────

def shot_main_window(app, save):
    """The whole window, Convert tab, as it opens."""
    from eubi_bridge.qt_gui.main_window import MainWindow
    window = MainWindow()
    window.resize(*WINDOW_SIZE)
    window.show()
    app.processEvents()
    save(window, "main-window")
    return window


def shot_parameter_tabs(app, save):
    """One image per parameter tab, so each can be referenced on its own."""
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    page = ConvertPage()
    page.resize(*WINDOW_SIZE)
    page.show()
    app.processEvents()

    for index in range(page._tabs.count()):
        title = page._tabs.tabText(index).strip().lower().replace(" ", "-")
        page._tabs.setCurrentIndex(index)
        app.processEvents()
        save(page, f"tab-{title}")
    return page


def shot_batch_queue(app, save):
    """The batch queue with rows, overrides, and the grouped header."""
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage, _MODE_BATCH, _LAST_TAB
    page = ConvertPage()
    page.resize(*WINDOW_SIZE)
    page.show()
    page._mode_bar.setCurrentIndex(_MODE_BATCH)
    page._tabs.setCurrentIndex(_LAST_TAB)
    _sample_batch(page)
    app.processEvents()
    save(page, "batch-queue")

    # The same queue showing every parameter, which is where the three-row
    # header and the category gutters earn their place.
    page._batch_full_table.setChecked(True)
    app.processEvents()
    save(page, "batch-queue-full-table")
    page._batch_full_table.setChecked(False)

    # And with one category expanded, the common middle ground.
    if getattr(page, "_batch_tab_toggles", None):
        first = next(iter(page._batch_tab_toggles.values()))
        first.setChecked(True)
        app.processEvents()
        save(page, "batch-queue-category-shown")
        first.setChecked(False)
    return page


def shot_edit_cells(app, save):
    """The Edit Cells dialog, including a greyed-out dependent parameter."""
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage, _MODE_BATCH, _LAST_TAB
    from eubi_bridge.qt_gui.widgets.batch_cell_editor import BatchCellEditor

    page = ConvertPage()
    page.resize(*WINDOW_SIZE)
    page._mode_bar.setCurrentIndex(_MODE_BATCH)
    page._tabs.setCurrentIndex(_LAST_TAB)
    _sample_batch(page)
    app.processEvents()

    # auto_chunk together with the manual sizes it governs, so the greying and
    # its explanation are both visible.
    keys = ["auto_chunk", "target_chunk_mb", "z_chunk", "dtype"]
    dialog = BatchCellEditor(page._batch, [0, 1], keys, parent=page)
    dialog.resize(*DIALOG_SIZE)
    dialog.show()
    app.processEvents()
    save(dialog, "edit-cells-dialog")
    dialog.close()
    return page


def shot_channel_colours(app, save):
    """The Metadata tab with a couple of channel colours overridden."""
    from eubi_bridge.qt_gui.pages.convert_page import ConvertPage
    page = ConvertPage()
    page.resize(*WINDOW_SIZE)
    page.show()

    for index in range(page._tabs.count()):
        if page._tabs.tabText(index).strip().lower() == "metadata":
            page._tabs.setCurrentIndex(index)
            break

    rows = page._channel_colour_rows
    if len(rows) >= 3:
        rows[0]["override"].setChecked(True)
        rows[0]["hex"] = "FF8800"
        rows[2]["override"].setChecked(True)
        rows[2]["hex"] = "00AAFF"
    app.processEvents()
    save(page, "channel-colours")
    return page


SHOTS = {
    "main": shot_main_window,
    "tabs": shot_parameter_tabs,
    "batch": shot_batch_queue,
    "edit-cells": shot_edit_cells,
    "colours": shot_channel_colours,
}


# ── driver ────────────────────────────────────────────────────────────────────

def _display(path: Path) -> str:
    """Path relative to the repository when it is inside it, else absolute."""
    try:
        return str(path.relative_to(_ROOT))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", choices=sorted(SHOTS),
                        help="generate only these groups (repeatable)")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR,
                        help=f"output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--list", action="store_true",
                        help="list the shot groups and exit")
    parser.add_argument("--allow-missing-fonts", action="store_true",
                        help="write images even if Qt cannot render text")
    args = parser.parse_args(argv)

    if args.list:
        for name, func in sorted(SHOTS.items()):
            summary = (func.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:12} {summary}")
        return 0

    _configure_qt()
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    if not _fonts_available(app) and not args.allow_missing_fonts:
        print(
            "No fonts are available to Qt, so every label would render as a "
            "box.\nInstall fonts (e.g. 'apt install fonts-dejavu-core') and "
            "run again,\nor pass --allow-missing-fonts to write the images "
            "anyway.",
            file=sys.stderr,
        )
        return 2

    _apply_appearance(app)

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(widget, name: str) -> None:
        path = args.out / f"{name}.png"
        if not widget.grab().save(str(path)):
            raise RuntimeError(f"could not write {path}")
        written.append(path)
        print(f"  {_display(path)}")

    selected = args.only or sorted(SHOTS)
    # Pages are kept alive until the end: ConvertPage owns background helpers
    # that complain when collected mid-run.
    alive = []
    for name in selected:
        print(f"{name}:")
        alive.append(SHOTS[name](app, save))

    print(f"\n{len(written)} image(s) written to {_display(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
