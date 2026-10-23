import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import square
from PyQt6 import QtWidgets, uic
from openpyxl import Workbook
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QMessageBox, QLineEdit, QFileDialog, QVBoxLayout, QWidget,
    QPushButton, QLabel, QCheckBox, QComboBox, QSlider
)
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

###############################################################################
# 1) CyclotronSimulation (unchanged)
###############################################################################
class CyclotronSimulation:
    def __init__(self):
        self.B = 0.185
        self.q = 1.602176462e-19
        self.m_p = 1.67262158e-27
        self.m_n = 0.0
        self.m = self.m_p + self.m_n
        self.U = 1000
        self.d = 20.0e-3
        self.f_HF_initial = 2.82e6
        self.use_square_wave = False
        self.phi_deg = 0

        self.t0 = 0.0
        self.t_end = 1.4e-6
        self.time_step = 1e-9

        self.chamber_radius = 0.07
        self.gap_width = 0.02
        self.margin = 0.003

        self.x0, self.y0 = -0.01, 0.0
        self.vx0, self.vy0 = 0.0, 0.0

        self.update_derived_parameters()

    def update_derived_parameters(self):
        self.Ey = self.U / self.d
        self.w_HF_initial = 2.0 * np.pi * self.f_HF_initial
        self.phi = np.deg2rad(self.phi_deg)
        self.m = self.m_p + self.m_n
        self.w_zyc = self.q / self.m * self.B
        self.time = np.arange(self.t0, self.t_end, self.time_step)

        self.top_flat = self.gap_width / 2
        self.bottom_flat = -(self.gap_width / 2)
        self.R = self.chamber_radius - self.margin

    def electric_field(self, t):
        E_sin = self.Ey * np.cos(self.w_HF_initial * t - self.phi)
        E_square = self.Ey * square(self.w_HF_initial * t - self.phi)
        return E_sin, E_square

    def odefun(self, r, t):
        x, y, vx, vy = r
        if -self.d/2 <= y <= self.d/2:
            E_sin, E_square = self.electric_field(t)
            E = E_square if self.use_square_wave else E_sin
        else:
            E = 0.0
        ax = self.w_zyc * vy
        ay = (self.q / self.m) * E - self.w_zyc * vx
        return [vx, vy, ax, ay]

    def run_simulation(self):
        initial_state = [self.x0, self.y0, self.vx0, self.vy0]
        self.solution = odeint(self.odefun, initial_state, self.time)
        self.x, self.y, self.vx, self.vy = self.solution.T
        self.z = np.sqrt(self.vx**2 + self.vy**2)
        self.r = np.sqrt(self.x**2 + self.y**2)

        self.s = np.zeros_like(self.x)
        for i in range(1, len(self.x)):
            dx = self.x[i] - self.x[i-1]
            dy = self.y[i] - self.y[i-1]
            self.s[i] = self.s[i-1] + np.sqrt(dx*dx + dy*dy)

        if np.any(self.r >= self.chamber_radius):
            self.exit_index = np.where(self.r >= self.chamber_radius)[0][0]
        else:
            self.exit_index = len(self.r)

        self.time_trimmed = self.time[:self.exit_index]
        self.x_trimmed = self.x[:self.exit_index]
        self.y_trimmed = self.y[:self.exit_index]
        self.vx_trimmed = self.vx[:self.exit_index]
        self.vy_trimmed = self.vy[:self.exit_index]
        self.z_trimmed = self.z[:self.exit_index]
        self.s_trimmed = self.s[:self.exit_index]

        self.solution_matrix = np.column_stack((
            self.time_trimmed, self.x_trimmed, self.y_trimmed,
            self.vx_trimmed, self.vy_trimmed, self.z_trimmed, self.s_trimmed
        ))

        entry_indices = np.where(self.y_trimmed >= self.gap_width / 2)[0]
        self.entry_index = entry_indices[0] if len(entry_indices) else 0

        self.velocity_initial_dee = self.z_trimmed[self.entry_index] if len(self.z_trimmed) else 0.0
        self.radius_initial_dee = (self.m * self.velocity_initial_dee / (self.q * self.B)) * 1000
        self.energy_initial_dee = 0.5 * self.m * self.velocity_initial_dee**2 / (self.q * 1000)

        self.velocity_final = self.z_trimmed[-1] if len(self.z_trimmed) else 0.0
        self.radius_final = (self.m * self.velocity_final / (self.q * self.B)) * 1000
        self.energy_final = 0.5 * self.m * self.velocity_final**2 / (self.q * 1000)

    def plot_trajectory(self, figure=None):
        import matplotlib.pyplot as plt
        if figure is None:
            plt.figure(figsize=(8,8))
            ax = plt.gca()
        else:
            figure.clear()
            ax = figure.add_subplot(111)

        theta_vals = np.linspace(0, 2*np.pi, 300)
        x_circ = self.chamber_radius * np.cos(theta_vals)
        y_circ = self.chamber_radius * np.sin(theta_vals)
        ax.plot(x_circ, y_circ, 'g-', linewidth=2, label='Chamber Boundary')

        ax.plot(self.x_trimmed, self.y_trimmed, 'k-', linewidth=1.5, label='Ion Trajectory')

        if self.R > 0:
            top_flat = self.gap_width / 2
            bottom_flat = -self.gap_width / 2
            theta0_top = np.arcsin(top_flat / self.R) if top_flat < self.R else 0.0
            theta_top_vals = np.linspace(theta0_top, np.pi - theta0_top, 100)
            x_top_arc = self.R * np.cos(theta_top_vals)
            y_top_arc = self.R * np.sin(theta_top_vals)
            ax.plot(x_top_arc, y_top_arc, 'r-', lw=2)
            corner_x = np.sqrt(self.R**2 - top_flat**2) if top_flat < self.R else 0
            ax.plot([-corner_x, corner_x], [top_flat, top_flat], 'r-', lw=2)

            theta0_bottom = np.arcsin(abs(bottom_flat)/self.R) if abs(bottom_flat) < self.R else 0.0
            theta_bottom_vals = np.linspace(np.pi + theta0_bottom, 2*np.pi - theta0_bottom, 100)
            x_bot_arc = self.R * np.cos(theta_bottom_vals)
            y_bot_arc = self.R * np.sin(theta_bottom_vals)
            ax.plot(x_bot_arc, y_bot_arc, 'r-', lw=2)
            corner_x2 = np.sqrt(self.R**2 - bottom_flat**2) if abs(bottom_flat) < self.R else 0
            ax.plot([-corner_x2, corner_x2], [bottom_flat, bottom_flat], 'r-', lw=2)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Ion Trajectory")
        ax.grid(True)
        ax.axis('equal')
        ax.set_xlim(-self.chamber_radius, self.chamber_radius)
        ax.set_ylim(-self.chamber_radius, self.chamber_radius)
        ax.legend()

        if figure is None:
            plt.show()

    def plot_electric_field(self, figure=None):
        import matplotlib.pyplot as plt
        if figure is None:
            plt.figure(figsize=(8,6))
            ax = plt.gca()
        else:
            figure.clear()
            ax = figure.add_subplot(111)

        E_sin_vals, E_square_vals = zip(*[self.electric_field(t) for t in self.time_trimmed])
        E_sin_vals = np.array(E_sin_vals)
        E_square_vals = np.array(E_square_vals)

        ax.plot(self.time_trimmed, E_sin_vals, 'b--', label='E-Feld (Sinusoidal)')
        ax.plot(self.time_trimmed, E_square_vals, 'r-', label='E-Feld (Square)')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("E (V/m)")
        ax.set_title("Electric Field vs Time")
        ax.grid(True)
        ax.legend()

        if figure is None:
            plt.show()

    def plot_additional(self):
        """Placeholder for 'plot additional' button."""
        print("Plotting additional diagrams (user can expand).")

###############################################################################
# 2) MplCanvas
###############################################################################
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

###############################################################################
# 3) MainWindow
###############################################################################
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cyclotron Simulation with Full UI")

        UI_FILE_PATH = r"C:\Users\user\Downloads\STUD\MY other\Internship\ZykloSim_nov24.ui"
        if not os.path.exists(UI_FILE_PATH):
            print(f"UI file not found: {UI_FILE_PATH}")
            sys.exit(1)

        # Load the .ui into self
        uic.loadUi(UI_FILE_PATH, self)

        # Force a style that overrides any higher-level stylesheet that might hide text
        # This line overrides color for QLineEdits, QLabels, QComboBoxes, etc.
        override_style = """
        QLineEdit {
            color: black;
            background-color: white;
            font-size: 14px;
        }
        QLabel {
            color: black;
            font-size: 14px;
        }
        QComboBox {
            color: black;
            background-color: white;
            font-size: 14px;
        }
        QCheckBox {
            color: black;
            font-size: 14px;
        }
        QRadioButton {
            color: black;
            font-size: 14px;
        }
        QPushButton {
            color: black;
            background-color: #E0E0E0;
            font-size: 14px;
        }
        """
        self.setStyleSheet(override_style)

        # Create the simulation
        self.simulation = CyclotronSimulation()

        # Insert MplCanvas in graphwidget
        self.graphwidget = self.findChild(QWidget, "graphwidget")
        self.canvas = MplCanvas(self, width=8, height=8, dpi=100)
        if self.graphwidget:
            layout = QVBoxLayout(self.graphwidget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.canvas)

        # Gather QLineEdits
        self.b_edit = self.findChild(QLineEdit, "b_edit")
        self.f_edit = self.findChild(QLineEdit, "f_edit")
        self.margin_edit = self.findChild(QLineEdit, "margin_edit")
        self.chamber_radius_edit = self.findChild(QLineEdit, "chamber_radius_edit")
        self.gap_width_edit = self.findChild(QLineEdit, "gap_width_edit")
        self.xo_edit = self.findChild(QLineEdit, "lineEdit")
        self.yo_edit = self.findChild(QLineEdit, "yo_edit")
        self.v_xo_edit = self.findChild(QLineEdit, "vxo_edit")
        self.v_yo_edit = self.findChild(QLineEdit, "vyo_edit")
        self.u_edit = self.findChild(QLineEdit, "u_edit")
        self.d_edit = self.findChild(QLineEdit, "d_edit")
        self.phi_edit = self.findChild(QLineEdit, "phi_edit")

        # Read-only fields
        self.anfangsgeschw_edit = self.findChild(QLineEdit, "vo_label")
        self.endgeschw_edit = self.findChild(QLineEdit, "v_end_label")
        self.anfangsenergie_edit = self.findChild(QLineEdit, "Eo_label")
        self.endenenergie_edit = self.findChild(QLineEdit, "E_end_label")
        self.anfangsradius_edit = self.findChild(QLineEdit, "ro_label")
        self.endradius_edit = self.findChild(QLineEdit, "r_end_label")
        self.bahnlange_edit = self.findChild(QLineEdit, "r_end_label_2")

        # Force them read-only with a gray background
        for ro in [
            self.anfangsgeschw_edit, self.endgeschw_edit, self.anfangsenergie_edit,
            self.endenenergie_edit, self.anfangsradius_edit, self.endradius_edit, self.bahnlange_edit
        ]:
            if ro:
                ro.setReadOnly(True)
                ro.setStyleSheet("QLineEdit { background-color: #F0F0F0; color: black; }")

        # Diagram combos, checkboxes
        self.diagram_combobox = self.findChild(QComboBox, "Diagramm_combobox")
        self.linestyle_combobox = self.findChild(QComboBox, "linestyle_combobox")
        self.linecolor_comboBox = self.findChild(QComboBox, "linecolor_comboBox")
        self.animation_checkBox = self.findChild(QCheckBox, "animation_checkBox")

        # Buttons
        self.start_Button = self.findChild(QPushButton, "start_Button")
        self.export_Button = self.findChild(QPushButton, "export_Button")
        self.loesch_Button = self.findChild(QPushButton, "loesch_Button")
        self.plot_button = self.findChild(QPushButton, "plot_button")

        # Sliders
        self.b_slider = self.findChild(QSlider, "b_Slider")
        self.f_slider = self.findChild(QSlider, "f_Slider")

        # e.g. freq label
        self.fzyk_label = self.findChild(QLabel, "fzyk_label")

        self._setup_validators()
        self._connect_signals()
        self._initialize_values()

        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Ready")
        self.status_bar_default_stylesheet = self.statusBar.styleSheet()
        self.statusBar.setFixedHeight(int(self.statusBar.height() * 0.75))

        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink_status_bar)
        self.is_dimmed = False

        self.resize(1400, 1200)

    def _setup_validators(self):
        dbl = QDoubleValidator()
        fields = [
            self.b_edit, self.f_edit, self.margin_edit,
            self.chamber_radius_edit, self.gap_width_edit, self.xo_edit,
            self.yo_edit, self.v_xo_edit, self.v_yo_edit,
            self.u_edit, self.d_edit, self.phi_edit
        ]
        for fld in fields:
            if fld:
                fld.setValidator(dbl)

    def _connect_signals(self):
        if self.start_Button:
            self.start_Button.clicked.connect(self.run_simulation)
        if self.export_Button:
            self.export_Button.clicked.connect(self.export_to_excel)
        if self.loesch_Button:
            self.loesch_Button.clicked.connect(self._loesch_button_clicked)
        if self.plot_button:
            self.plot_button.clicked.connect(self._handle_plot_additional)

        if self.diagram_combobox:
            self.diagram_combobox.currentIndexChanged.connect(self._replot_with_current_style)
        if self.linestyle_combobox:
            self.linestyle_combobox.currentIndexChanged.connect(self._replot_with_current_style)
        if self.linecolor_comboBox:
            self.linecolor_comboBox.currentIndexChanged.connect(self._replot_with_current_style)
        if self.animation_checkBox:
            self.animation_checkBox.stateChanged.connect(self._handle_animation_toggle)

        if self.b_slider:
            self.b_slider.valueChanged.connect(self._update_b_input_from_slider)
        if self.f_slider:
            self.f_slider.valueChanged.connect(self._update_f_input_from_slider)

    def _initialize_values(self):
        """Fills each QLineEdit with default from simulation."""
        if self.b_edit:
            self.b_edit.setText(f"{self.simulation.B:.3f}")
        if self.f_edit:
            self.f_edit.setText(f"{self.simulation.f_HF_initial/1e6:.2f}")
        if self.margin_edit:
            self.margin_edit.setText(f"{self.simulation.margin}")
        if self.chamber_radius_edit:
            self.chamber_radius_edit.setText(f"{self.simulation.chamber_radius}")
        if self.gap_width_edit:
            self.gap_width_edit.setText(f"{self.simulation.gap_width}")
        if self.xo_edit:
            self.xo_edit.setText(f"{self.simulation.x0}")
        if self.yo_edit:
            self.yo_edit.setText(f"{self.simulation.y0}")
        if self.v_xo_edit:
            self.v_xo_edit.setText(f"{self.simulation.vx0}")
        if self.v_yo_edit:
            self.v_yo_edit.setText(f"{self.simulation.vy0}")
        if self.u_edit:
            self.u_edit.setText(f"{self.simulation.U}")
        if self.d_edit:
            self.d_edit.setText(f"{self.simulation.d}")
        if self.phi_edit:
            self.phi_edit.setText(f"{self.simulation.phi_deg}")
        if self.fzyk_label:
            self.fzyk_label.setText("0.00 MHz")

        # read-only fields to default 0
        if self.anfangsgeschw_edit:
            self.anfangsgeschw_edit.setText("0.0")
        if self.endgeschw_edit:
            self.endgeschw_edit.setText("0.0")
        if self.anfangsenergie_edit:
            self.anfangsenergie_edit.setText("0.0")
        if self.endenenergie_edit:
            self.endenenergie_edit.setText("0.0")
        if self.anfangsradius_edit:
            self.anfangsradius_edit.setText("0.0")
        if self.endradius_edit:
            self.endradius_edit.setText("0.0")
        if self.bahnlange_edit:
            self.bahnlange_edit.setText("0.0")

    def _update_b_input_from_slider(self):
        if self.b_slider and self.b_edit:
            val = self.b_slider.value()/100.0
            self.b_edit.setText(f"{val:.3f}")

    def _update_f_input_from_slider(self):
        if self.f_slider and self.f_edit:
            val = self.f_slider.value()/100.0
            self.f_edit.setText(f"{val:.2f}")

    def run_simulation(self):
        """Reads user inputs, runs simulation, updates read-only fields."""
        try:
            self._read_inputs_into_simulation()
            self.simulation.update_derived_parameters()
            self.simulation.run_simulation()

            self._plot_main()

            # Show results
            if self.anfangsgeschw_edit:
                self.anfangsgeschw_edit.setText(f"{self.simulation.velocity_initial_dee:.4g}")
            if self.endgeschw_edit:
                self.endgeschw_edit.setText(f"{self.simulation.velocity_final:.4g}")
            if self.anfangsenergie_edit:
                self.anfangsenergie_edit.setText(f"{self.simulation.energy_initial_dee:.2f}")
            if self.endenenergie_edit:
                self.endenenergie_edit.setText(f"{self.simulation.energy_final:.2f}")
            if self.anfangsradius_edit:
                self.anfangsradius_edit.setText(f"{self.simulation.radius_initial_dee:.4f}")
            if self.endradius_edit:
                self.endradius_edit.setText(f"{self.simulation.radius_final:.4f}")
            if len(self.simulation.s_trimmed):
                self.bahnlange_edit.setText(f"{self.simulation.s_trimmed[-1]:.4f}")
            else:
                self.bahnlange_edit.setText("0.0")

            # cyclotron freq in MHz
            f_zyc = (self.simulation.w_zyc/(2*np.pi))/1e6 if self.simulation.w_zyc!=0 else 0
            if self.fzyk_label:
                self.fzyk_label.setText(f"{f_zyc:.2f} MHz")

            self.statusBar.showMessage("Simulation completed", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error: {e}")
            self.statusBar.showMessage(f"Error: {e}", 5000)

    def _plot_main(self):
        line_style, line_color = self._get_style_from_combos()
        diag = self._get_diagram_choice()

        self.canvas.axes.clear()

        if diag == "x-y":
            self.simulation.plot_trajectory(figure=self.canvas.fig)
        elif diag == "t-v":
            self._plot_t_v(line_style, line_color)
        elif diag == "Phase":
            # Show electric field vs time
            self.simulation.plot_electric_field(figure=self.canvas.fig)
        elif diag == "t-x":
            self._plot_t_x(line_style, line_color)
        elif diag == "t-y":
            self._plot_t_y(line_style, line_color)
        else:
            self.simulation.plot_trajectory(figure=self.canvas.fig)

        self.canvas.draw()

    def _handle_animation_toggle(self, state):
        if state == Qt.CheckState.Checked:
            self.statusBar.showMessage("Animation On (example).")
        else:
            self.statusBar.showMessage("Animation Off.")

    def _plot_t_v(self, line_style, line_color):
        ax = self.canvas.fig.add_subplot(111)
        t = self.simulation.time_trimmed
        v = self.simulation.z_trimmed
        ax.plot(t, v, linestyle=line_style, color=line_color, label='|v| (m/s)')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Time vs Velocity")
        ax.grid(True)
        ax.legend()

    def _plot_t_x(self, line_style, line_color):
        ax = self.canvas.fig.add_subplot(111)
        t = self.simulation.time_trimmed
        x = self.simulation.x_trimmed
        ax.plot(t, x, linestyle=line_style, color=line_color, label='x(t)')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("X (m)")
        ax.set_title("Time vs X")
        ax.grid(True)
        ax.legend()

    def _plot_t_y(self, line_style, line_color):
        ax = self.canvas.fig.add_subplot(111)
        t = self.simulation.time_trimmed
        y = self.simulation.y_trimmed
        ax.plot(t, y, linestyle=line_style, color=line_color, label='y(t)')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Time vs Y")
        ax.grid(True)
        ax.legend()

    def _get_style_from_combos(self):
        line_style = "-"
        line_color = "k"

        if self.linestyle_combobox:
            chosen_style = self.linestyle_combobox.currentText()
            style_map = {
                "----": "-",
                "- - - -": "--",
                ". . . .": ":",
                "- . -": "-.",
                "- . . -": (0,(5,5,1,5))
            }
            if chosen_style in style_map:
                line_style = style_map[chosen_style]

        if self.linecolor_comboBox:
            chosen_color = self.linecolor_comboBox.currentText()
            color_map = {
                "schwarz": "k",
                "rot": "r",
                "grün": "g",
                "blau": "b"
            }
            if chosen_color in color_map:
                line_color = color_map[chosen_color]

        return line_style, line_color

    def _get_diagram_choice(self):
        if not self.diagram_combobox:
            return "x-y"
        return self.diagram_combobox.currentText()

    def _read_inputs_into_simulation(self):
        try:
            self.simulation.B = float(self.b_edit.text())
        except:
            self.simulation.B = 0.185
        try:
            freq_mhz = float(self.f_edit.text())
            self.simulation.f_HF_initial = freq_mhz*1e6
        except:
            self.simulation.f_HF_initial = 2.82e6
        try:
            self.simulation.margin = float(self.margin_edit.text())
        except:
            self.simulation.margin = 0.003
        try:
            self.simulation.chamber_radius = float(self.chamber_radius_edit.text())
        except:
            self.simulation.chamber_radius = 0.07
        try:
            self.simulation.gap_width = float(self.gap_width_edit.text())
        except:
            self.simulation.gap_width = 0.02
        try:
            self.simulation.x0 = float(self.xo_edit.text())
        except:
            self.simulation.x0 = -0.01
        try:
            self.simulation.y0 = float(self.yo_edit.text())
        except:
            self.simulation.y0 = 0.0
        try:
            self.simulation.vx0 = float(self.v_xo_edit.text())
        except:
            self.simulation.vx0 = 0.0
        try:
            self.simulation.vy0 = float(self.v_yo_edit.text())
        except:
            self.simulation.vy0 = 0.0
        try:
            self.simulation.U = float(self.u_edit.text())
        except:
            self.simulation.U = 1000
        try:
            self.simulation.d = float(self.d_edit.text())
        except:
            self.simulation.d = 20.0e-3
        try:
            self.simulation.phi_deg = float(self.phi_edit.text())
        except:
            self.simulation.phi_deg = 0.0

    def export_to_excel(self):
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx)")
            if not file_path:
                return
            wb = Workbook()
            ws = wb.active
            ws.title = "Simulation Data"

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ws.append(["Export Timestamp", timestamp])
            ws.append([])
            headers = ["Parameter", "Value"]
            ws.append(headers)

            data = [
                ["B (T)", self.b_edit.text() if self.b_edit else ""],
                ["f (MHz)", self.f_edit.text() if self.f_edit else ""],
                ["Margin (m)", self.margin_edit.text() if self.margin_edit else ""],
                ["Chamber Radius (m)", self.chamber_radius_edit.text() if self.chamber_radius_edit else ""],
                ["Gap Width (m)", self.gap_width_edit.text() if self.gap_width_edit else ""],
                ["x0 (m)", self.xo_edit.text() if self.xo_edit else ""],
                ["y0 (m)", self.yo_edit.text() if self.yo_edit else ""],
                ["vx0 (m/s)", self.v_xo_edit.text() if self.v_xo_edit else ""],
                ["vy0 (m/s)", self.v_yo_edit.text() if self.v_yo_edit else ""],
                ["U (V)", self.u_edit.text() if self.u_edit else ""],
                ["d (m)", self.d_edit.text() if self.d_edit else ""],
                ["phi (deg)", self.phi_edit.text() if self.phi_edit else ""],
                ["Initial Velocity (m/s)", self.anfangsgeschw_edit.text() if self.anfangsgeschw_edit else ""],
                ["Final Velocity (m/s)", self.endgeschw_edit.text() if self.endgeschw_edit else ""],
                ["Initial Energy (keV)", self.anfangsenergie_edit.text() if self.anfangsenergie_edit else ""],
                ["Final Energy (keV)", self.endenenergie_edit.text() if self.endenenergie_edit else ""],
                ["Initial Radius (mm)", self.anfangsradius_edit.text() if self.anfangsradius_edit else ""],
                ["Final Radius (mm)", self.endradius_edit.text() if self.endradius_edit else ""],
                ["Trajectory Length (m)", self.bahnlange_edit.text() if self.bahnlange_edit else ""],
            ]
            for row in data:
                ws.append(row)

            wb.save(file_path)
            QMessageBox.information(self, "Export Successful", f"Data exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting to Excel: {e}")

    def _loesch_button_clicked(self):
        """Clear fields, reset simulation."""
        for field in [
            self.b_edit, self.f_edit, self.margin_edit,
            self.chamber_radius_edit, self.gap_width_edit,
            self.xo_edit, self.yo_edit, self.v_xo_edit, self.v_yo_edit,
            self.u_edit, self.d_edit, self.phi_edit
        ]:
            if field:
                field.clear()

        if self.fzyk_label:
            self.fzyk_label.setText("0.00 MHz")

        for ro_field in [
            self.anfangsgeschw_edit, self.endgeschw_edit, self.anfangsenergie_edit,
            self.endenenergie_edit, self.anfangsradius_edit, self.endradius_edit,
            self.bahnlange_edit
        ]:
            if ro_field:
                ro_field.clear()

        self.statusBar.setStyleSheet(self.status_bar_default_stylesheet)
        self.statusBar.showMessage("Ready")
        self.blink_timer.stop()

        self.simulation.__init__()
        self._initialize_values()

        self.canvas.axes.clear()
        self.canvas.draw()

    def _handle_plot_additional(self):
        """If user clicks the 'plot_button'."""
        self.simulation.plot_additional()

    def blink_status_bar(self):
        dim_red = "QColor(255, 100, 100)"
        if hasattr(self, 'is_dimmed') and self.is_dimmed:
            self.statusBar.setStyleSheet("background-color: lightblue;")
        else:
            self.statusBar.setStyleSheet(f"background-color: {dim_red};")
        self.is_dimmed = not getattr(self, 'is_dimmed', False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
