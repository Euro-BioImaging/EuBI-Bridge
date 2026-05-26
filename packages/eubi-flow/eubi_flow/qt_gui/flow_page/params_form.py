"""Dynamic parameter form generated from a processor's Params Pydantic model.

Introspects ``processor.params_model.model_fields`` to build the correct
widget for each field type.  Inline validation highlights invalid fields in
red and disables the parent panel's Apply button.
"""
from __future__ import annotations

import typing
from typing import Any

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

_AXES = "tczyx"
_BORDER_OK  = ""
_BORDER_ERR = "border: 1px solid #e53935;"


def _get_constraints(field_info) -> dict[str, Any]:
    """Extract ge/le/gt/lt from Pydantic v2 field metadata."""
    result: dict[str, Any] = {}
    try:
        import annotated_types
        for meta in field_info.metadata:
            if isinstance(meta, annotated_types.Ge):
                result["ge"] = meta.ge
            elif isinstance(meta, annotated_types.Le):
                result["le"] = meta.le
            elif isinstance(meta, annotated_types.Gt):
                result["gt"] = meta.gt
            elif isinstance(meta, annotated_types.Lt):
                result["lt"] = meta.lt
    except ImportError:
        pass
    return result


def _literal_choices(annotation) -> list[str] | None:
    if typing.get_origin(annotation) is typing.Literal:
        return [str(a) for a in typing.get_args(annotation)]
    return None


def _is_filter_wave(processor) -> bool:
    """Filter waves have extra='allow' and a sigma or size base field."""
    try:
        extra = getattr(processor.params_model, "model_config", {}).get("extra")
        fields = processor.params_model.model_fields
        return extra == "allow" and ("sigma" in fields or "size" in fields)
    except Exception:
        return False


class ParamsForm(QWidget):
    """Form widget generated from a processor's Params model.

    Signals
    -------
    params_changed(dict)
        Emitted whenever any widget changes value (after inline validation).
    validation_changed(bool)
        Emitted with ``True`` when the form is valid, ``False`` when not.
    """

    params_changed   = pyqtSignal(dict)
    validation_changed = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._widgets: dict[str, QWidget] = {}
        self._axis_tabs: dict[str, QTabWidget] = {}
        self._processor_name: str | None = None
        self._valid = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_wave(self, wave_spec) -> None:
        """Rebuild the form for *wave_spec*."""
        self.clear()
        self._processor_name = wave_spec.name

        try:
            from eubi_flow.registry import get_processor
            processor = get_processor(wave_spec.name)
        except Exception:
            lbl = QLabel(f"Unknown processor: {wave_spec.name}")
            self._layout.addWidget(lbl)
            return

        if processor.params_model is None:
            lbl = QLabel("(no parameters)")
            lbl.setStyleSheet("color: #888888;")
            self._layout.addWidget(lbl)
            return

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        fields = processor.params_model.model_fields
        params = wave_spec.params

        for field_name, field_info in fields.items():
            annotation = field_info.annotation
            choices = _literal_choices(annotation)
            constraints = _get_constraints(field_info)
            current = params.get(field_name, field_info.default)
            required = field_info.is_required()

            if choices is not None:
                widget = QComboBox()
                widget.addItems(choices)
                if current is not None and str(current) in choices:
                    widget.setCurrentText(str(current))
                widget.currentTextChanged.connect(self._on_change)

            elif annotation is bool or (hasattr(annotation, "__origin__") is False
                                         and annotation is bool):
                widget = QCheckBox()
                widget.setChecked(bool(current) if current is not None else False)
                widget.toggled.connect(self._on_change)

            elif annotation is int:
                widget = QSpinBox()
                widget.setRange(int(constraints.get("ge", constraints.get("gt", 0))),
                                int(constraints.get("le", constraints.get("lt", 99999))))
                if current is not None:
                    widget.setValue(int(current))
                widget.valueChanged.connect(self._on_change)

            elif annotation is float:
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setSingleStep(0.1)
                ge = constraints.get("ge", constraints.get("gt"))
                le = constraints.get("le", constraints.get("lt"))
                widget.setRange(
                    float(ge) if ge is not None else -1e9,
                    float(le) if le is not None else 1e9,
                )
                if current is not None:
                    widget.setValue(float(current))
                if required and current is None:
                    widget.setStyleSheet(_BORDER_ERR)
                widget.valueChanged.connect(self._on_change)

            else:
                from PyQt6.QtWidgets import QLineEdit
                widget = QLineEdit()
                widget.setText(str(current) if current is not None else "")
                widget.textChanged.connect(self._on_change)

            self._widgets[field_name] = widget
            label_text = field_name + (" *" if required else "")
            form.addRow(label_text, widget)

        wrapper = QWidget()
        wrapper.setLayout(form)
        self._layout.addWidget(wrapper)

        # Per-axis overrides for filter waves
        if _is_filter_wave(processor):
            base_key = "sigma" if "sigma" in fields else "size"
            self._add_axis_section(base_key, params)

        self._layout.addStretch(1)
        self._run_validation()

    def clear(self) -> None:
        while self._layout.count():
            child = self._layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._widgets.clear()
        self._axis_tabs.clear()
        self._processor_name = None

    def collect_params(self) -> dict:
        """Return current form values as a flat dict."""
        result: dict[str, Any] = {}
        for name, w in self._widgets.items():
            result[name] = _read_widget(w)
        # Per-axis overrides
        for base_key, tab in self._axis_tabs.items():
            per_axis_tab = tab.widget(1)
            if per_axis_tab is None:
                continue
            for ax in _AXES:
                child = per_axis_tab.findChild(QDoubleSpinBox, f"{ax}_{base_key}")
                if child is not None and child.value() != 0.0:
                    result[f"{ax}_{base_key}"] = child.value()
        return {k: v for k, v in result.items() if v is not None}

    # ------------------------------------------------------------------

    def _add_axis_section(self, base_key: str, params: dict) -> None:
        group = QGroupBox("Per-axis overrides")
        group.setCheckable(True)
        group.setChecked(any(f"{ax}_{base_key}" in params for ax in _AXES))
        v = QVBoxLayout(group)

        tab = QTabWidget()
        tab.setTabPosition(QTabWidget.TabPosition.North)

        # Uniform tab (mirrors the base widget, already in form)
        uniform_placeholder = QLabel("See field above.")
        uniform_placeholder.setStyleSheet("color: #888888;")
        tab.addTab(uniform_placeholder, "Uniform")

        # Per-axis tab
        per_tab = QWidget()
        per_form = QFormLayout(per_tab)
        per_form.setSpacing(4)
        for ax in _AXES:
            sb = QDoubleSpinBox()
            sb.setObjectName(f"{ax}_{base_key}")
            sb.setRange(0.0, 1e6)
            sb.setDecimals(2)
            sb.setSingleStep(0.5)
            sb.setSpecialValueText("—")  # 0 = use uniform
            current = params.get(f"{ax}_{base_key}", 0.0)
            sb.setValue(float(current))
            sb.valueChanged.connect(self._on_change)
            per_form.addRow(f"{ax}:", sb)
        tab.addTab(per_tab, "Per-axis")

        v.addWidget(tab)
        self._layout.addWidget(group)
        self._axis_tabs[base_key] = tab

    def _on_change(self, *args) -> None:
        self._run_validation()

    def _run_validation(self) -> None:
        if self._processor_name is None:
            return
        try:
            from eubi_flow.registry import get_processor
            processor = get_processor(self._processor_name)
            errors = processor.validate_params(self.collect_params())
        except Exception:
            errors = []

        # Build a set of offending field names
        err_fields = set()
        for msg in errors:
            # msg format: "field_name: message"
            if ":" in msg:
                err_fields.add(msg.split(":")[0].strip())

        for fname, widget in self._widgets.items():
            if fname in err_fields:
                widget.setStyleSheet(_BORDER_ERR)
                widget.setToolTip(next((m for m in errors if m.startswith(fname)), ""))
            else:
                widget.setStyleSheet(_BORDER_OK)
                widget.setToolTip("")

        valid = len(errors) == 0
        if valid != self._valid:
            self._valid = valid
            self.validation_changed.emit(valid)

        if valid:
            self.params_changed.emit(self.collect_params())

    @property
    def is_valid(self) -> bool:
        return self._valid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_widget(w: QWidget) -> Any:
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return w.value()
    from PyQt6.QtWidgets import QLineEdit
    if isinstance(w, QLineEdit):
        return w.text()
    return None
