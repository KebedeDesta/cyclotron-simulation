from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure

from PyQt6 import QtGui, QtWidgets, uic
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSignalBlocker, QSize, QTimer, Qt
from PyQt6.QtGui import (
    QAction,
    QDoubleValidator,
    QIntValidator,
    QKeySequence,
    QPainter,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cyclotron_sim import (
    CyclotronConfig,
    CyclotronSimulation,
    InstrumentConfig,
    Particle,
    measure_beam,
)
from cyclotron_sim.instrumentation import MeasurementResult
from cyclotron_sim.model import SimulationResult
from cyclotron_sim.plots import (
    chamber_geometry,
    diagnostics_figure,
    instrumentation_figure,
)
from cyclotron_sim.report import format_report, write_excel

UI_FILE = Path(__file__).resolve().with_name("ZykloSim_nov24.ui")

LINE_STYLES = {
    "----": "-",
    "- - - -": "--",
    ". . . .": ":",
    "- . -": "-.",
    "- . . -": (0, (5, 5, 1, 5)),
}
LINE_COLORS = ("#111111", "#dc2626", "#16a34a", "#2563eb")


class FitView(QGraphicsView):
    BASE_WIDTH = 1600
    BASE_HEIGHT = 880

    def __init__(self, widget: QWidget, parent=None) -> None:
        self.scene = QGraphicsScene(parent)
        self.proxy = self.scene.addWidget(widget)
        super().__init__(self.scene, parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setInteractive(True)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QtGui.QColor("#e8ebef"))
        self.canvas = None
        self.graph = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_widget()

    def fit_widget(self) -> None:
        if self.proxy.widget() is None:
            return
        viewport = self.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        if (
            viewport.width() >= self.BASE_WIDTH
            and viewport.height() >= self.BASE_HEIGHT
        ):
            target_width = viewport.width()
            target_height = viewport.height()
            scale = 1.0
        else:
            target_width = self.BASE_WIDTH
            target_height = self.BASE_HEIGHT
            scale = min(
                viewport.width() / self.BASE_WIDTH,
                viewport.height() / self.BASE_HEIGHT,
            )
        self.proxy.setGeometry(
            QRectF(0.0, 0.0, float(target_width), float(target_height))
        )
        self.setSceneRect(self.proxy.sceneBoundingRect())
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(self.proxy)

    def wheelEvent(self, event) -> None:
        if self.canvas is not None and self.graph is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            local = self.proxy.mapFromScene(scene_pos)
            if self.graph.geometry().contains(local.toPoint()):
                canvas_pos = self.canvas.mapFrom(
                    self.proxy.widget(), local.toPoint()
                )
                forwarded = QWheelEvent(
                    QPointF(canvas_pos),
                    event.globalPosition(),
                    event.pixelDelta(),
                    event.angleDelta(),
                    event.buttons(),
                    event.modifiers(),
                    event.phase(),
                    event.inverted(),
                )
                QtWidgets.QApplication.sendEvent(self.canvas, forwarded)
                event.accept()
                return
        super().wheelEvent(event)


class DiagnosticsDialog(QDialog):
    def __init__(
        self,
        result: SimulationResult,
        measurement: MeasurementResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Instrumentation diagnostics")
        self.resize(1180, 900)

        layout = QVBoxLayout(self)
        figure = instrumentation_figure(result, measurement, figsize=(10.5, 6.8))
        canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(canvas, stretch=3)

        report = QPlainTextEdit()
        report.setReadOnly(True)
        report.setPlainText(format_report(result, measurement))
        report.setFont(QtGui.QFont("Consolas", 9))
        report.setMaximumHeight(240)
        layout.addWidget(report, stretch=1)

        beam_figure = diagnostics_figure(result, figsize=(10.5, 6.0))
        beam_canvas = FigureCanvasQTAgg(beam_figure)
        beam_canvas.setMaximumHeight(280)
        layout.addWidget(beam_canvas, stretch=2)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        if not UI_FILE.exists():
            raise FileNotFoundError(f"UI file not found: {UI_FILE}")

        uic.loadUi(str(UI_FILE), self)
        self.setWindowTitle("ZykloSim - cyclotron measurement simulation")

        central = self.takeCentralWidget()
        if central is None:
            central = self.centralWidget()
        central.setMinimumSize(1600, 880)
        self.central_panel = central
        self.view = FitView(central, self)
        self.setCentralWidget(self.view)
        self.setMinimumSize(900, 620)
        self._size_to_screen()

        self._syncing = False
        self._animation = None
        self._result: SimulationResult | None = None
        self._measurement: MeasurementResult | None = None
        self._last_config = CyclotronConfig()
        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(350)
        self._auto_timer.timeout.connect(self._auto_run)

        self._find_widgets()
        self._build_canvas()
        self._setup_validators()
        self._setup_sliders()
        self._connect_signals()
        if self.animation_checkbox:
            self.animation_checkbox.setChecked(True)
        self._apply_config_to_ui(self._last_config)
        self._clear_readouts()
        self._resize_graph()
        self.central_panel.installEventFilter(self)
        self.view.canvas = self.canvas
        self.view.graph = self.graphwidget
        self.view.fit_widget()
        self.statusBar().showMessage(
            "Ready - mouse wheel zooms, double-click resets the plot"
        )

    def _size_to_screen(self) -> None:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1600, 900)
            return
        available = screen.availableGeometry()
        width = min(1640, max(1000, available.width() - 60))
        height = min(940, max(680, available.height() - 80))
        self.resize(width, height)

    def eventFilter(self, watched, event) -> bool:
        if (
            getattr(self, "central_panel", None) is watched
            and event.type() == QEvent.Type.Resize
        ):
            self._resize_graph()
        return super().eventFilter(watched, event)

    def _resize_graph(self) -> None:
        graph = getattr(self, "graphwidget", None)
        panel = getattr(self, "central_panel", None)
        if graph is None or panel is None:
            return
        width = max(800, panel.width() - 780)
        height = max(600, panel.height() - 140)
        if graph.width() != width or graph.height() != height:
            graph.setGeometry(760, 10, width, height)
        slider_y = 10 + height + 40
        if getattr(self, "f_slider", None):
            self.f_slider.setGeometry(760, slider_y, width, 41)
        for label, fraction in getattr(self, "_f_label_positions", []):
            geometry = label.geometry()
            label.setGeometry(
                760 + int(round(fraction * width)),
                slider_y + 30,
                geometry.width(),
                geometry.height(),
            )

    def _find_widgets(self) -> None:
        root = self.central_panel
        self.q_edit = root.findChild(QLineEdit, "q_edit")
        self.p_edit = root.findChild(QLineEdit, "p_edit")
        self.n_edit = root.findChild(QLineEdit, "n_edit")
        self.b_edit = root.findChild(QLineEdit, "b_edit")
        self.f_edit = root.findChild(QLineEdit, "f_edit")
        self.u_edit = root.findChild(QLineEdit, "u_edit")
        self.d_edit = root.findChild(QLineEdit, "d_edit")
        self.phi_edit = root.findChild(QLineEdit, "phi_edit")
        self.x0_edit = root.findChild(QLineEdit, "lineEdit")
        self.y0_edit = root.findChild(QLineEdit, "yo_edit")
        self.vx0_edit = root.findChild(QLineEdit, "vxo_edit")
        self.vy0_edit = root.findChild(QLineEdit, "vyo_edit")
        self.cup_edit = root.findChild(QLineEdit, "x_faraday_edit")
        self.reduction_edit = root.findChild(QLineEdit, "Faktor_edit")

        self.waveform_combo = root.findChild(QComboBox, "U_comboBox")
        self.cup_checkbox = root.findChild(QCheckBox, "visible_checkBox")
        self.diagram_combo = root.findChild(QComboBox, "Diagramm_combobox")
        self.linestyle_combo = root.findChild(QComboBox, "linestyle_combobox")
        self.linecolor_combo = root.findChild(QComboBox, "linecolor_comboBox")
        self.animation_checkbox = root.findChild(
            QCheckBox, "animation_checkBox"
        )

        self.vo_label = root.findChild(QLabel, "vo_label")
        self.eo_label = root.findChild(QLabel, "Eo_label")
        self.ro_label = root.findChild(QLabel, "ro_label")
        self.v_end_label = root.findChild(QLabel, "v_end_label")
        self.e_end_label = root.findChild(QLabel, "E_end_label")
        self.r_end_label = root.findChild(QLabel, "r_end_label")
        self.path_label = root.findChild(QLabel, "r_end_label_2")
        self.fzyk_label = root.findChild(QLabel, "fzyk_label")

        self.start_button = root.findChild(QPushButton, "start_Button")
        self.export_button = root.findChild(QPushButton, "export_Button")
        self.clear_button = root.findChild(QPushButton, "loesch_Button")
        self.plot_button = root.findChild(QPushButton, "plot_button")

        self.b_slider = root.findChild(QSlider, "b_Slider")
        self.f_slider = root.findChild(QSlider, "f_Slider")
        self.f_slider_labels = [
            root.findChild(QLabel, name)
            for name in (
                "f_label_5",
                "f_label_10",
                "f_label_11",
                "f_label_20",
            )
        ]
        self._f_label_positions = [
            (label, (label.geometry().x() - 760) / 801.0)
            for label in self.f_slider_labels
            if label
        ]
        self.graphwidget = root.findChild(QWidget, "graphwidget")
        if self.graphwidget:
            self.graphwidget.setToolTip(
                "Mouse wheel: zoom at cursor; double-click: reset; "
                "toolbar: pan, box zoom, save"
            )

        self.cup_edit.setToolTip("Faraday cup radial position in mm")
        self.reduction_edit.setToolTip(
            "Reduction of the nominal acceleration voltage in percent"
        )
        self.cup_checkbox.setToolTip(
            "Insert the Faraday cup; the beam stops at the selected radius"
        )

    def _build_canvas(self) -> None:
        self.figure = Figure(figsize=(7.0, 6.2), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout = QVBoxLayout(self.graphwidget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(QSize(18, 18))
        self.toolbar.setMaximumHeight(30)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_scroll(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        factor = 1.25 if event.button == "up" else 1.0 / 1.25
        self._zoom_axes(event.inaxes, factor, event.xdata, event.ydata)

    def _on_click(self, event) -> None:
        if event.dblclick and event.inaxes is not None:
            self._reset_view()

    def _zoom_axes(self, axes, factor: float, xdata: float, ydata: float) -> None:
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        if x_span <= 0 or y_span <= 0:
            return
        x_fraction = (xdata - xlim[0]) / x_span
        y_fraction = (ydata - ylim[0]) / y_span
        new_x_span = x_span / factor
        new_y_span = y_span / factor
        axes.set_xlim(
            xdata - x_fraction * new_x_span,
            xdata + (1.0 - x_fraction) * new_x_span,
        )
        axes.set_ylim(
            ydata - y_fraction * new_y_span,
            ydata + (1.0 - y_fraction) * new_y_span,
        )
        self.canvas.draw_idle()

    def _zoom_in(self) -> None:
        if self._result is None:
            return
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self._zoom_axes(
            self.axes,
            1.35,
            0.5 * (xlim[0] + xlim[1]),
            0.5 * (ylim[0] + ylim[1]),
        )

    def _zoom_out(self) -> None:
        if self._result is None:
            return
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self._zoom_axes(
            self.axes,
            1.0 / 1.35,
            0.5 * (xlim[0] + xlim[1]),
            0.5 * (ylim[0] + ylim[1]),
        )

    def _reset_view(self) -> None:
        if self._result is None:
            return
        self._stop_animation()
        self._plot_main()
        self.statusBar().showMessage("View reset", 2000)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _setup_validators(self) -> None:
        double_validator = QDoubleValidator()
        double_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        integer_validator = QIntValidator()
        for widget in (
            self.b_edit,
            self.f_edit,
            self.u_edit,
            self.d_edit,
            self.phi_edit,
            self.x0_edit,
            self.y0_edit,
            self.vx0_edit,
            self.vy0_edit,
            self.cup_edit,
            self.reduction_edit,
        ):
            if widget:
                widget.setValidator(double_validator)
        for widget in (self.q_edit, self.p_edit, self.n_edit):
            if widget:
                widget.setValidator(integer_validator)

    def _setup_sliders(self) -> None:
        if self.b_slider:
            self.b_slider.setMinimum(0)
            self.b_slider.setMaximum(1000)
            self.b_slider.setValue(185)
        if self.f_slider:
            self.f_slider.setMinimum(0)
            self.f_slider.setMaximum(2000)
            self.f_slider.setValue(282)

    def _connect_signals(self) -> None:
        if self.start_button:
            self.start_button.clicked.connect(lambda: self.run_simulation())
        if self.export_button:
            self.export_button.clicked.connect(self.export_to_excel)
        if self.clear_button:
            self.clear_button.clicked.connect(self.reset)
        if self.plot_button:
            self.plot_button.clicked.connect(self.show_diagnostics)

        if self.b_slider:
            self.b_slider.valueChanged.connect(self._b_slider_changed)
        if self.f_slider:
            self.f_slider.valueChanged.connect(self._f_slider_changed)
        if self.b_edit:
            self.b_edit.textChanged.connect(self._b_text_changed)
        if self.f_edit:
            self.f_edit.textChanged.connect(self._f_text_changed)

        if self.diagram_combo:
            self.diagram_combo.currentIndexChanged.connect(self._replot)
        if self.linestyle_combo:
            self.linestyle_combo.currentIndexChanged.connect(self._replot)
        if self.linecolor_combo:
            self.linecolor_combo.currentIndexChanged.connect(self._replot)
        if self.animation_checkbox:
            self.animation_checkbox.stateChanged.connect(self._animation_toggled)

        for widget in (
            self.q_edit,
            self.p_edit,
            self.n_edit,
            self.u_edit,
            self.d_edit,
            self.phi_edit,
            self.x0_edit,
            self.y0_edit,
            self.vx0_edit,
            self.vy0_edit,
            self.cup_edit,
            self.reduction_edit,
        ):
            if widget:
                widget.textChanged.connect(self._field_changed)
        if self.waveform_combo:
            self.waveform_combo.currentIndexChanged.connect(self._field_changed)
        if self.cup_checkbox:
            self.cup_checkbox.stateChanged.connect(self._field_changed)

        actions = {
            "actionKonfiguration_laden": self.load_configuration,
            "actionKonfiguration_speichern": self.save_configuration,
            "actionExport_nach_Excel": self.export_to_excel,
            "actionBeenden": self.close,
        }
        for name, slot in actions.items():
            action = self.findChild(QAction, name)
            if action:
                action.triggered.connect(slot)
        diagram_actions = {
            "actionx_y": "x-y",
            "actiont_v": "t-v",
            "actionPhase": "Phase",
            "actiont_x": "t-x",
            "actiont_y": "t-y",
        }
        for name, text in diagram_actions.items():
            action = self.findChild(QAction, name)
            if action:
                action.triggered.connect(
                    lambda _checked=False, value=text: self._select_diagram(value)
                )
        self._build_view_menu()

    def _build_view_menu(self) -> None:
        bar = getattr(self, "menuBar", None)
        if not isinstance(bar, QtWidgets.QMenuBar):
            bar = self.menuBar()
        menu = bar.addMenu("Ansicht")
        zoom_in = QAction("Zoom +", self)
        zoom_in.setShortcut(QKeySequence("Ctrl+="))
        zoom_in.triggered.connect(self._zoom_in)
        menu.addAction(zoom_in)
        zoom_out = QAction("Zoom -", self)
        zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out.triggered.connect(self._zoom_out)
        menu.addAction(zoom_out)
        reset = QAction("Ansicht zuruecksetzen", self)
        reset.setShortcut(QKeySequence("Ctrl+0"))
        reset.triggered.connect(self._reset_view)
        menu.addAction(reset)
        menu.addSeparator()
        fit = QAction("Fenster an Bildschirm anpassen", self)
        fit.setShortcut(QKeySequence("Ctrl+F"))
        fit.triggered.connect(self._fit_window)
        menu.addAction(fit)
        fullscreen = QAction("Vollbild", self)
        fullscreen.setShortcut(QKeySequence("F11"))
        fullscreen.triggered.connect(self._toggle_fullscreen)
        menu.addAction(fullscreen)
        for action in (zoom_in, zoom_out, reset, fit, fullscreen):
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)

        self._view_shortcuts = [QShortcut(QKeySequence("Ctrl++"), self)]
        for shortcut in self._view_shortcuts:
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(self._zoom_in)

    def _fit_window(self) -> None:
        self.showMaximized()
        self.view.fit_widget()
        self.statusBar().showMessage("Window fitted to screen", 2000)

    def _select_diagram(self, text: str) -> None:
        if self.diagram_combo:
            index = self.diagram_combo.findText(text)
            if index >= 0:
                self.diagram_combo.setCurrentIndex(index)

    def _field_changed(self, *_args) -> None:
        if self._syncing:
            return
        self.statusBar().showMessage("Parameters changed - updating...")
        self._auto_timer.start()

    def _auto_run(self) -> None:
        if self._syncing:
            return
        self.run_simulation(silent=True)

    def _parse_float(self, widget: QLineEdit | None, default: float) -> float:
        if widget is None:
            return default
        text = widget.text().strip().replace(",", ".")
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def _parse_int(self, widget: QLineEdit | None, default: int) -> int:
        if widget is None:
            return default
        text = widget.text().strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default

    def _read_config(self) -> CyclotronConfig:
        particle = Particle(
            charge_number=self._parse_int(self.q_edit, 1),
            protons=self._parse_int(self.p_edit, 1),
            neutrons=self._parse_int(self.n_edit, 0),
        )
        waveform = "sine"
        if self.waveform_combo and self.waveform_combo.currentText().lower().startswith(
            "rechteck"
        ):
            waveform = "square"
        config = CyclotronConfig(
            particle=particle,
            magnetic_field=self._parse_float(self.b_edit, 0.185),
            rf_frequency=self._parse_float(self.f_edit, 2.82) * 1.0e6,
            acceleration_voltage=self._parse_float(self.u_edit, 1000.0),
            gap_width=self._parse_float(self.d_edit, 0.02),
            rf_phase_deg=self._parse_float(self.phi_edit, 0.0),
            waveform=waveform,
            x0=self._parse_float(self.x0_edit, -0.01),
            y0=self._parse_float(self.y0_edit, 0.0),
            vx0=self._parse_float(self.vx0_edit, 0.0),
            vy0=self._parse_float(self.vy0_edit, 0.0),
            voltage_reduction_pct=self._parse_float(self.reduction_edit, 0.0),
            cup_enabled=bool(self.cup_checkbox and self.cup_checkbox.isChecked()),
            cup_radius_mm=self._parse_float(self.cup_edit, 32.0),
            t_end=4.0e-6,
            time_step=1.0e-9,
        )
        minimum_time = min(30.0 / config.cyclotron_frequency, 20.0e-6)
        config.t_end = max(config.t_end, minimum_time)
        return config

    def _apply_config_to_ui(self, config: CyclotronConfig) -> None:
        self._syncing = True
        try:
            values = {
                self.q_edit: str(config.particle.charge_number),
                self.p_edit: str(config.particle.protons),
                self.n_edit: str(config.particle.neutrons),
                self.b_edit: f"{config.magnetic_field:.3f}",
                self.f_edit: f"{config.rf_frequency / 1.0e6:.2f}",
                self.u_edit: f"{config.acceleration_voltage:.1f}",
                self.d_edit: f"{config.gap_width:.3f}",
                self.phi_edit: f"{config.rf_phase_deg:.1f}",
                self.x0_edit: f"{config.x0:.4f}",
                self.y0_edit: f"{config.y0:.4f}",
                self.vx0_edit: f"{config.vx0:.3f}",
                self.vy0_edit: f"{config.vy0:.3f}",
                self.cup_edit: f"{config.cup_radius_mm:.2f}",
                self.reduction_edit: f"{config.voltage_reduction_pct:.1f}",
            }
            for widget, text in values.items():
                if widget:
                    widget.setText(text)
            if self.waveform_combo:
                self.waveform_combo.setCurrentText(
                    "Rechteck" if config.waveform == "square" else "Sinus"
                )
            if self.cup_checkbox:
                self.cup_checkbox.setChecked(config.cup_enabled)
            if self.b_slider:
                self.b_slider.setValue(int(round(config.magnetic_field * 1000)))
            if self.f_slider:
                self.f_slider.setValue(
                    int(round(config.rf_frequency / 1.0e6 * 100))
                )
            self._last_config = config
        finally:
            self._syncing = False

    def _b_slider_changed(self, value: int) -> None:
        if self._syncing or not self.b_edit:
            return
        self._syncing = True
        self.b_edit.setText(f"{value / 1000.0:.3f}")
        self._syncing = False
        self._field_changed()

    def _f_slider_changed(self, value: int) -> None:
        if self._syncing or not self.f_edit:
            return
        self._syncing = True
        self.f_edit.setText(f"{value / 100.0:.2f}")
        self._syncing = False
        self._field_changed()

    def _b_text_changed(self, text: str) -> None:
        if self._syncing or not self.b_slider:
            return
        try:
            value = int(round(float(text.replace(",", ".")) * 1000))
        except ValueError:
            return
        self._syncing = True
        self.b_slider.setValue(max(0, min(value, 1000)))
        self._syncing = False
        self._field_changed()

    def _f_text_changed(self, text: str) -> None:
        if self._syncing or not self.f_slider:
            return
        try:
            value = int(round(float(text.replace(",", ".")) * 100))
        except ValueError:
            return
        self._syncing = True
        self.f_slider.setValue(max(0, min(value, 2000)))
        self._syncing = False
        self._field_changed()

    def _line_style(self) -> str | tuple:
        if self.linestyle_combo:
            return LINE_STYLES.get(self.linestyle_combo.currentText(), "-")
        return "-"

    def _line_color(self) -> str:
        if self.linecolor_combo:
            index = self.linecolor_combo.currentIndex()
            if 0 <= index < len(LINE_COLORS):
                return LINE_COLORS[index]
        return LINE_COLORS[0]

    def run_simulation(self, silent: bool = False) -> None:
        self._auto_timer.stop()
        try:
            config = self._read_config()
        except ValueError as error:
            if silent:
                self.statusBar().showMessage(f"Invalid parameters: {error}", 4000)
            else:
                QMessageBox.critical(self, "Invalid parameters", str(error))
            return

        self._stop_animation()
        self.statusBar().showMessage("Simulating...")
        QtWidgets.QApplication.processEvents()
        try:
            simulation = CyclotronSimulation(config)
            result = simulation.run()
            measurement = measure_beam(result, InstrumentConfig())
        except Exception as error:
            if silent:
                self.statusBar().showMessage(
                    f"Simulation error: {error}", 4000
                )
            else:
                QMessageBox.critical(self, "Simulation error", str(error))
                self.statusBar().showMessage(f"Error: {error}", 5000)
            return

        self._result = result
        self._measurement = measurement
        self._last_config = config
        self._update_readouts(result, measurement)
        self._plot_main()
        if self.animation_checkbox and self.animation_checkbox.isChecked():
            self._start_animation()

        frequency = measurement.pickup_frequency_hz
        if np.isfinite(frequency):
            frequency_text = f"{frequency / 1.0e6:.4f} MHz"
        else:
            frequency_text = "not resolved"
        prefix = "Live update" if silent else result.exit_reason
        self.statusBar().showMessage(
            f"{prefix}: {result.n_turns} turns, "
            f"E = {result.final_energy_kev:.3f} keV, "
            f"f_pickup = {frequency_text}"
        )

    def _update_readouts(
        self, result: SimulationResult, measurement: MeasurementResult
    ) -> None:
        labels = {
            self.vo_label: f"{result.initial_speed:.4g}",
            self.v_end_label: f"{result.final_speed:.4g}",
            self.eo_label: f"{result.initial_energy_j / 1.602176634e-16:.3f}",
            self.e_end_label: f"{result.final_energy_kev:.3f}",
            self.ro_label: f"{result.initial_radius_mm:.3f}",
            self.r_end_label: f"{result.final_radius_mm:.3f}",
            self.path_label: f"{result.total_path_length:.4f}",
            self.fzyk_label: f"{result.config.cyclotron_frequency / 1.0e6:.3f} MHz",
        }
        for label, text in labels.items():
            if label:
                label.setText(text)
        if self.fzyk_label:
            self.fzyk_label.setToolTip(
                f"Measured pickup: {measurement.pickup_frequency_hz / 1.0e6:.4f} MHz"
            )

    def _clear_readouts(self) -> None:
        for label in (
            self.vo_label,
            self.eo_label,
            self.ro_label,
            self.v_end_label,
            self.e_end_label,
            self.r_end_label,
            self.path_label,
        ):
            if label:
                label.setText("0.0")
        if self.fzyk_label:
            self.fzyk_label.setText("0.000 MHz")

    def _plot_main(self) -> None:
        if self._result is None:
            return
        result = self._result
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        diagram = (
            self.diagram_combo.currentText() if self.diagram_combo else "x-y"
        )
        color = self._line_color()
        style = self._line_style()

        if diagram == "t-v":
            self.axes.plot(
                result.t * 1.0e6,
                result.speed / 1.0e3,
                color=color,
                linestyle=style,
            )
            self.axes.set_xlabel("time (us)")
            self.axes.set_ylabel("speed (km/s)")
            self.axes.set_title("Beam velocity")
            self.axes.grid(True, alpha=0.25)
        elif diagram == "Phase":
            self.axes.plot(
                result.x * 1.0e3,
                result.vx / 1.0e3,
                color=color,
                linestyle=style,
            )
            self.axes.set_xlabel("x (mm)")
            self.axes.set_ylabel("vx (km/s)")
            self.axes.set_title("Phase portrait")
            self.axes.grid(True, alpha=0.25)
        elif diagram == "t-x":
            self.axes.plot(
                result.t * 1.0e6,
                result.x * 1.0e3,
                color=color,
                linestyle=style,
            )
            self.axes.set_xlabel("time (us)")
            self.axes.set_ylabel("x (mm)")
            self.axes.set_title("Horizontal position")
            self.axes.grid(True, alpha=0.25)
        elif diagram == "t-y":
            self.axes.plot(
                result.t * 1.0e6,
                result.y * 1.0e3,
                color=color,
                linestyle=style,
            )
            self.axes.set_xlabel("time (us)")
            self.axes.set_ylabel("y (mm)")
            self.axes.set_title("Vertical position")
            self.axes.grid(True, alpha=0.25)
        else:
            chamber_geometry(self.axes, result)
            self.axes.plot(
                result.x,
                result.y,
                color=color,
                linestyle=style,
                linewidth=1.8,
                label="ion trajectory",
            )
            self.axes.set_title("Beam trajectory")
            self.axes.legend(loc="upper right", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _replot(self, *_args) -> None:
        if self._result is None:
            return
        if self.animation_checkbox and self.animation_checkbox.isChecked():
            diagram = (
                self.diagram_combo.currentText() if self.diagram_combo else "x-y"
            )
            if diagram != "x-y":
                self.animation_checkbox.setChecked(False)
                return
            self._start_animation()
        else:
            self._plot_main()

    def _animation_toggled(self, state: int) -> None:
        if state == Qt.CheckState.Checked.value:
            if self.diagram_combo:
                blocker = QSignalBlocker(self.diagram_combo)
                index = self.diagram_combo.findText("x-y")
                if index >= 0:
                    self.diagram_combo.setCurrentIndex(index)
                del blocker
            self._start_animation()
        else:
            self._stop_animation()
            self._plot_main()

    def _start_animation(self) -> None:
        if self._result is None:
            return
        self._stop_animation()
        result = self._result
        diagram = (
            self.diagram_combo.currentText() if self.diagram_combo else "x-y"
        )
        if diagram != "x-y":
            if self.diagram_combo:
                blocker = QSignalBlocker(self.diagram_combo)
                index = self.diagram_combo.findText("x-y")
                if index >= 0:
                    self.diagram_combo.setCurrentIndex(index)
                del blocker
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        chamber_geometry(self.axes, result)
        color = self._line_color()
        (line,) = self.axes.plot(
            [], [], color=color, linewidth=1.8, label="ion trajectory"
        )
        (marker,) = self.axes.plot(
            [], [], "o", color=color, markersize=5, label="ion"
        )
        self.axes.set_title("Beam trajectory")
        self.axes.legend(loc="upper right", fontsize=8)
        self.figure.tight_layout()
        stride = max(1, result.t.size // 240)
        frames = list(range(1, result.t.size, stride))
        if frames and frames[-1] != result.t.size - 1:
            frames.append(result.t.size - 1)

        def update(frame_index: int):
            line.set_data(result.x[:frame_index], result.y[:frame_index])
            marker.set_data(
                [result.x[frame_index - 1]], [result.y[frame_index - 1]]
            )
            return line, marker

        from matplotlib.animation import FuncAnimation

        self._animation = FuncAnimation(
            self.figure,
            update,
            frames=frames,
            interval=20,
            blit=False,
            repeat=True,
            cache_frame_data=False,
        )
        self.canvas.draw_idle()
        self.statusBar().showMessage("Animation running")

    def _stop_animation(self) -> None:
        if self._animation is not None:
            self._animation.event_source.stop()
            self._animation = None

    def show_diagnostics(self) -> None:
        if self._result is None or self._measurement is None:
            QMessageBox.information(
                self, "No data", "Run a simulation first."
            )
            return
        dialog = DiagnosticsDialog(
            self._result, self._measurement, parent=self
        )
        dialog.exec()

    def export_to_excel(self) -> None:
        if self._result is None or self._measurement is None:
            QMessageBox.information(
                self, "No data", "Run a simulation first."
            )
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export simulation", "cyclotron_results.xlsx", "Excel (*.xlsx)"
        )
        if not file_name:
            return
        try:
            path = write_excel(self._result, self._measurement, file_name)
            report_path = Path(path).with_suffix(".txt")
            report_path.write_text(
                format_report(self._result, self._measurement),
                encoding="utf-8",
            )
            QMessageBox.information(
                self,
                "Export complete",
                f"Saved:\n{path}\n{report_path}",
            )
        except Exception as error:
            QMessageBox.critical(self, "Export error", str(error))

    def save_configuration(self) -> None:
        try:
            config = self._read_config()
        except ValueError as error:
            QMessageBox.critical(self, "Invalid parameters", str(error))
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", "cyclotron_config.json", "JSON (*.json)"
        )
        if not file_name:
            return
        Path(file_name).write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )
        self.statusBar().showMessage(f"Configuration saved to {file_name}", 4000)

    def load_configuration(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load configuration", "", "JSON (*.json)"
        )
        if not file_name:
            return
        try:
            data = json.loads(Path(file_name).read_text(encoding="utf-8"))
            particle = Particle(**data.pop("particle"))
            config = CyclotronConfig(particle=particle, **data)
        except Exception as error:
            QMessageBox.critical(self, "Configuration error", str(error))
            return
        self._apply_config_to_ui(config)
        self.statusBar().showMessage(f"Configuration loaded from {file_name}", 4000)

    def reset(self) -> None:
        self._stop_animation()
        self._result = None
        self._measurement = None
        self._apply_config_to_ui(CyclotronConfig())
        self._clear_readouts()
        if self.animation_checkbox:
            self.animation_checkbox.setChecked(False)
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.canvas.draw_idle()
        self.statusBar().showMessage("Ready")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        window = MainWindow()
    except Exception as error:
        QMessageBox.critical(None, "Startup error", str(error))
        return 1
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
