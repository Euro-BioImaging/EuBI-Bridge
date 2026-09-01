"""
Edit Cells dialog: bulk-edit parameters across selected batch rows.

Usage:
    dlg = BatchCellEditor(model, rows=[0, 2], keys=["dtype"], parent=page)
    if dlg.exec():
        dlg.apply()     # writes straight into the BatchModel

The dialog describes parameters through :class:`~eubi_bridge.qt_gui.core.batch.
ParamSpec` rather than reusing the conversion form's widgets: those are single
hard-wired instances owned by the page, so the editor generates its own controls
from the spec table instead.

Two behaviours follow from how the batch CSV is read:

* A blank cell means "use the config value", and setting a field back to that
  value clears the override on its own.  Each field therefore carries a reset
  button rather than a separate mode: it fills in the config value, which the
  user would otherwise have to know and retype.
* When the selected rows disagree on a value, the editor starts blank and
  untouched fields are left alone, so a bulk edit never silently flattens
  differences the user could not see.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eubi_bridge.qt_gui.core.batch import (
    BatchModel, ParamSpec, grouped_specs, spec_for, to_cell, _PARAM_SPECS)


# exec() result meaning "re-open with one more parameter", distinct from
# Accepted/Rejected so the caller can tell a rebuild from a real confirmation.
_ADD_PARAMETER = 2


class _Field(QWidget):
    """One parameter row: an editor plus a reset-to-config button.

    The reset button puts the config's own value back, which clears the
    override because a value equal to the config renders as a blank cell.
    ``changed`` stays False until the user actually touches something, so a
    dialog opened over disagreeing rows leaves untouched fields alone.
    """

    def __init__(self, spec: ParamSpec, value, agreed: bool,
                 config_value=None, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.changed = False
        self.inactive = False
        self.config_value = config_value

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.editor = self._build_editor(spec, value if agreed else None)
        lay.addWidget(self.editor, 1)

        # Setting a field back to the config's value already clears the override
        # (values_equal makes the cell blank again), so this is a shortcut, not a
        # separate mode: it fills in a number the user would otherwise have to
        # know and retype.
        self.reset_btn = QPushButton("↺")
        self.reset_btn.setFixedWidth(26)
        self.reset_btn.setToolTip(
            f"Set back to the config value ({to_cell(config_value)}), "
            "which clears the override.")
        self.reset_btn.clicked.connect(self._on_reset_to_config)
        lay.addWidget(self.reset_btn)
        # A path is the row's identity, not an override: there is no config
        # value behind it, and a blank path would make the row unrunnable.
        if spec.kind in ("file", "directory") or config_value is None:
            self.reset_btn.setVisible(False)

        self._apply_tooltip()
        self._mixed = not agreed

    def _apply_tooltip(self):
        """Field tooltip: its own help plus the config value behind it."""
        parts = []
        if self.spec.tooltip:
            parts.append(self.spec.tooltip)
        if self.config_value is not None:
            parts.append(f"Config value: {to_cell(self.config_value)}")
        self.editor.setToolTip("\n".join(parts))

    def _on_reset_to_config(self):
        """Put the config's value back into the editor."""
        self.set_value(self.config_value)
        self._touch()

    # -- construction --

    def _build_editor(self, spec: ParamSpec, value) -> QWidget:
        if spec.kind == "bool":
            w = QComboBox()
            w.addItems(["True", "False"])
            if value is not None:
                w.setCurrentText("True" if value else "False")
            else:
                w.setCurrentIndex(-1)
            w.currentIndexChanged.connect(self._touch)
            return w

        if spec.kind == "choice":
            w = QComboBox()
            w.addItems(list(spec.choices))
            if value is not None and str(value) in spec.choices:
                w.setCurrentText(str(value))
            else:
                w.setCurrentIndex(-1)
            w.currentIndexChanged.connect(self._touch)
            return w

        if spec.kind in ("file", "directory"):
            # A line edit plus Browse: paths are edited far more often than any
            # other cell, and retyping one by hand is slow and error prone.
            w = QWidget()
            row = QHBoxLayout(w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            w.line = QLineEdit()
            if value is not None:
                w.line.setText(str(value))
            w.line.textEdited.connect(self._touch)
            row.addWidget(w.line, 1)
            browse = QPushButton("Browse...")
            browse.setFixedWidth(80)
            browse.clicked.connect(lambda: self._browse(w.line, spec.kind))
            row.addWidget(browse)
            return w

        if spec.kind == "auto_int":
            # A number, or 'auto' meaning "compute it".  Distinct from a blank
            # cell, which falls back to the config value instead.
            w = QSpinBox()
            w.setRange(int(spec.minimum) - 1, int(spec.maximum))
            w.setSpecialValueText("auto")
            w.setValue(w.minimum())
            if value is not None and str(value).strip().lower() != "auto":
                try:
                    w.setValue(int(value))
                except (TypeError, ValueError):
                    pass
            w.valueChanged.connect(self._touch)
            return w

        if spec.kind == "int":
            w = QSpinBox()
            w.setRange(int(spec.minimum), int(spec.maximum))
            if value is not None:
                try:
                    w.setValue(int(value))
                except (TypeError, ValueError):
                    pass
            w.valueChanged.connect(self._touch)
            return w

        if spec.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(float(spec.minimum), float(spec.maximum))
            if value is not None:
                try:
                    w.setValue(float(value))
                except (TypeError, ValueError):
                    pass
            w.valueChanged.connect(self._touch)
            return w

        w = QLineEdit()
        if value is not None:
            w.setText(str(value))
        w.textEdited.connect(self._touch)
        return w

    # -- state --

    def set_inactive(self, inactive: bool, reason: str = "") -> None:
        """Disable this field because its parent switch makes it inert.

        The user did not choose the config value here; the parameter simply
        does not apply.  :meth:`BatchCellEditor.apply` clears such overrides so
        the row cannot carry a value that looks applied but is ignored at write
        time.
        """
        self.inactive = inactive
        self.editor.setEnabled(not inactive)
        self.reset_btn.setEnabled(not inactive)
        if inactive and reason:
            self.editor.setToolTip(
                f"Not used: {reason} makes this inactive. "
                "Any override is cleared.")
        else:
            self._apply_tooltip()

    def _browse(self, line, kind: str):
        """Pick a path, starting from whatever the field already holds."""
        start = line.text().strip()
        if kind == "directory":
            chosen = QFileDialog.getExistingDirectory(
                self, "Select output directory", start)
        else:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Select input file", start)
            if not chosen:
                # An OME-Zarr store is a directory, so offer that too rather
                # than making zarr inputs unpickable.
                chosen = QFileDialog.getExistingDirectory(
                    self, "Select input directory or OME-Zarr store", start)
        if chosen:
            line.setText(chosen)
            self._touch()

    def _touch(self, *_):
        self.changed = True

    def value(self):
        """The editor's value, converted to what the model should store."""
        w = self.editor
        if self.spec.kind in ("file", "directory"):
            return w.line.text().strip()
        if isinstance(w, QComboBox):
            text = w.currentText()
            if self.spec.kind == "bool":
                return text == "True"
            return text
        if isinstance(w, QSpinBox):
            if self.spec.kind == "auto_int" and w.value() == w.minimum():
                return "auto"
            return int(w.value())
        if isinstance(w, QDoubleSpinBox):
            return float(w.value())
        return w.text().strip()

    def set_value(self, value) -> None:
        """Put *value* into the editor, whatever widget kind it is."""
        w = self.editor
        if self.spec.kind in ("file", "directory"):
            w.line.setText("" if value is None else str(value))
        elif isinstance(w, QComboBox):
            text = ("True" if value else "False")                 if self.spec.kind == "bool" else str(value)
            index = w.findText(text)
            w.setCurrentIndex(index)
        elif isinstance(w, QSpinBox):
            if self.spec.kind == "auto_int" and (
                    value is None or str(value).strip().lower() == "auto"):
                w.setValue(w.minimum())          # the 'auto' position
            else:
                try:
                    w.setValue(int(value))
                except (TypeError, ValueError):
                    pass
        elif isinstance(w, QDoubleSpinBox):
            try:
                w.setValue(float(value))
            except (TypeError, ValueError):
                pass
        else:
            w.setText("" if value is None else str(value))

    def is_blank(self) -> bool:
        """True when nothing is selected/entered, so there is nothing to write."""
        w = self.editor
        if self.spec.kind in ("file", "directory"):
            return not w.line.text().strip()
        if isinstance(w, QComboBox):
            return w.currentIndex() < 0
        if isinstance(w, QLineEdit):
            return not w.text().strip()
        return False


class BatchCellEditor(QDialog):
    """Bulk-edit the given parameter *keys* across the given *rows*."""

    def __init__(self, model: BatchModel, rows: list[int], keys: list[str],
                 parent=None):
        super().__init__(parent)
        self._model = model
        self._rows = sorted(set(rows))
        self._fields: dict[str, _Field] = {}

        self.setWindowTitle("Edit Cells")
        self.setMinimumWidth(460)

        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        header = QLabel()
        header.setStyleSheet("font-weight: bold;")
        lay.addWidget(header)

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(8)

        # Mirror the conversion form's own hierarchy: tab, then group box within
        # it, so a parameter appears exactly where the user is used to finding
        # it, however small the selection.
        mixed: list[str] = []
        for tab, groups in grouped_specs(keys):
            tab_box = QGroupBox(tab)
            tab_lay = QVBoxLayout(tab_box)
            tab_lay.setSpacing(6)

            for group, specs in groups:
                if group:
                    holder = QGroupBox(group)
                    form = QFormLayout(holder)
                else:
                    # Loose on the tab, exactly as on the form: no inner box.
                    holder = QWidget()
                    form = QFormLayout(holder)
                    form.setContentsMargins(0, 0, 0, 0)
                form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                form.setSpacing(6)

                for spec in specs:
                    value, agreed = model.common_value(self._rows, spec.key)
                    field = _Field(spec, value, agreed,
                                   config_value=model.config_value(spec.key))
                    self._fields[spec.key] = field
                    form.addRow(spec.label + ":", field)
                    if not agreed:
                        mixed.append(spec.label)
                tab_lay.addWidget(holder)

            body_lay.addWidget(tab_box)
        body_lay.addStretch(1)

        self._wire_dependencies()

        header.setText(
            f"Editing {len(self._fields)} parameter(s) "
            f"across {len(self._rows)} row(s)."
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)

        if mixed:
            note = QLabel(
                "Blank where the selected rows differ (" + ", ".join(mixed) +
                "). Leave a field untouched to keep those differences."
            )
            note.setWordWrap(True)
            note.setStyleSheet("font-size: 10px; color: #ffb74d;")
            lay.addWidget(note)

        # A parameter with no column yet is unreachable through cell selection,
        # notably any the batch has never overridden.  Offer the rest here so the
        # dialog is not limited to what the queue already shows.
        remaining = [spec for spec in _PARAM_SPECS
                     if spec.key not in self._fields]
        if remaining:
            add_row = QHBoxLayout()
            add_row.addWidget(QLabel("Add parameter:"))
            self._add_combo = QComboBox()
            self._add_combo.addItem("", "")
            for spec in remaining:
                label = f"{spec.tab}: {spec.label}" if spec.tab else spec.label
                self._add_combo.addItem(label, spec.key)
            self._add_combo.setToolTip(
                "Edit a parameter the queue does not show a column for yet.")
            self._add_combo.currentIndexChanged.connect(self._on_add_parameter)
            add_row.addWidget(self._add_combo, 1)
            lay.addLayout(add_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _on_add_parameter(self, index: int):
        """Re-open the dialog with the chosen parameter included."""
        key = self._add_combo.itemData(index)
        if not key or key in self._fields:
            return
        self.done(_ADD_PARAMETER)
        self.added_key = key

    # -- dependencies --

    def _wire_dependencies(self):
        """Grey out fields whose parent switch makes them inert.

        Mirrors the conversion form, where turning Auto chunk on hides the
        manual per-axis sizes.  The parent may not itself be in the dialog, in
        which case its value comes from the selected rows.
        """
        for key, field in self._fields.items():
            parent_key = field.spec.depends_on
            if not parent_key:
                continue
            parent_field = self._fields.get(parent_key)
            if parent_field is None:
                continue
            # The parent may be a bool or a choice, so pick whichever change
            # signal its editor actually has.
            editor = parent_field.editor
            for signal_name in ("currentIndexChanged", "valueChanged",
                                "textEdited"):
                signal = getattr(editor, signal_name, None)
                if signal is not None:
                    signal.connect(
                        lambda *_, k=parent_key: self._sync_dependants(k))
                    break
        for parent_key in {f.spec.depends_on for f in self._fields.values()
                           if f.spec.depends_on}:
            self._sync_dependants(parent_key)

    def _parent_value(self, parent_key: str):
        """Current value of *parent_key*: the dialog's field, else the rows'."""
        field = self._fields.get(parent_key)
        if field is not None and not field.is_blank():
            return field.value()
        value, agreed = self._model.common_value(self._rows, parent_key)
        return value if agreed else None

    def _sync_dependants(self, parent_key: str):
        parent_value = self._parent_value(parent_key)
        for key, field in self._fields.items():
            spec = field.spec
            if spec.depends_on != parent_key:
                continue
            if parent_value is None:
                # The selected rows disagree about the parent, so the parameter
                # is live for some of them: stay editable rather than guessing.
                field.set_inactive(False)
                continue
            if isinstance(spec.active_when, bool):
                active = bool(parent_value) == spec.active_when
            else:
                active = str(parent_value) == str(spec.active_when)
            # A parent that is itself inert cannot make this one live.
            parent_field = self._fields.get(parent_key)
            if parent_field is not None and parent_field.inactive:
                active = False
            if active:
                field.set_inactive(False)
            else:
                if isinstance(spec.active_when, bool):
                    reason = f"{parent_key}={not spec.active_when}"
                else:
                    reason = f"{parent_key}!={spec.active_when}"
                field.set_inactive(True, reason)
            # Re-evaluate anything hanging off this field, so a chain settles.
            if any(f.spec.depends_on == key for f in self._fields.values()):
                self._sync_dependants(key)

    # -- result --

    def apply(self) -> list[str]:
        """Write every touched field into the model; returns the keys changed."""
        applied: list[str] = []
        for key, field in self._fields.items():
            # An inert parameter must not keep an override: the value would show
            # as a column and read as applied while the writer ignores it.
            # Inertness is per row, and a mixed selection leaves the field
            # editable, so stale overrides are cleared independently of whether
            # the field itself was greyed out.
            stale = [i for i in self._rows
                     if self._model.is_inert(self._model.rows[i], key)
                     and self._model.cell(self._model.rows[i], key) is not None]
            if stale:
                self._model.reset_cells(stale, key)
                applied.append(key)
            if field.inactive or not field.changed:
                continue
            if field.is_blank():
                continue
            self._model.update_cells(self._rows, key, field.value())
            if key not in applied:
                applied.append(key)
        return applied
