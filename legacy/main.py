import sys
import os
from PyQt6 import QtWidgets, uic
from openpyxl import Workbook
import Draw_circle_nov24 as dc
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QPalette, QColor, QValidator
from PyQt6.QtWidgets import QMessageBox, QLineEdit, QFileDialog
import datetime  # Import datetime for timestamps


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        ui_file_path = r"C:\Users\user\PycharmProjects\pythonProject\venv\ZykloSim_nov24.ui"
        if not os.path.exists(ui_file_path):
            print(f"UI file not found at: {ui_file_path}")
            sys.exit(1)

        try:
            self.ui = uic.loadUi(ui_file_path)
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(self.ui)
            self.setCentralWidget(scroll_area)
            self.ui.setMinimumSize(1600, 1200)
        except Exception as e:
            print(f"Error loading UI file: {e}")
            sys.exit(1)

        # Define the shared slider style sheet
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: green;
                height: 8px;
                border-radius: 4px;
                margin: 2px 0;
            }
            QSlider::sub-page:horizontal {
                background: #bff;
                border: 1px solid #777;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #eee;
                border: 1px solid #777;
                width: 16px;
                margin: -5px 0;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #fff;
                border: 1px solid #777;
                height: 8px;
                border-radius: 4px;
            }
        """

        # Find and configure b_Slider (values between 0.00 and 1.00)
        self.b_slider = self.findChild(QtWidgets.QSlider, "b_Slider")
        if self.b_slider:
            self.b_slider.setMinimum(0)
            self.b_slider.setMaximum(100)  # 100/100 = 1.00
            self.b_slider.setStyleSheet(slider_style)
        else:
            print("Error: b_Slider not found.")
            sys.exit(1)

        try:
            self._setup_ui_elements()
        except AttributeError as e:
            print(f"Error accessing UI elements: {e}")
            sys.exit(1)

        # Configure f_Slider (values between 0.00 and 20.00)
        if self.f_slider:
            self.f_slider.setMinimum(0)
            self.f_slider.setMaximum(2000)  # 2000/100 = 20.00
            self.f_slider.setStyleSheet(slider_style)
        else:
            print("Error: f_Slider not found.")
            sys.exit(1)

        try:
            self.graphwidget = self.findChild(QtWidgets.QWidget, "graphwidget")
            if self.graphwidget:
                self.kammer = dc.kammerplot(self.graphwidget, 0.2, 0.7)
                self.kreis = dc.drawcircle(self.graphwidget, 0.5)
            else:
                print("Error: 'graphwidget' not found in the UI file.")
        except Exception as e:
            print(f"Error initializing graphics: {e}")
            sys.exit(1)

        self.resize(800, 600)

        # Create Status Bar
        self.statusBar = self.statusBar()  # Get the status bar
        self.statusBar.showMessage("Ready")
        self.status_bar_default_stylesheet = self.statusBar.styleSheet()  # Store the default stylesheet

        # Set Status Bar height to 50% of its default height.
        # This requires getting the default height first, which is tricky.
        # A simple, but less precise, way is to just set a fixed height.

        self.statusBar.setFixedHeight(int(self.statusBar.height() * 0.75))


        # Blinking Error
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.blink_status_bar)
        self.is_dimmed = False

        self._connect_signals()
        self._initialize_values()

    def _setup_ui_elements(self):
        self.q_edit = self.findChild(QtWidgets.QLineEdit, "q_edit")
        self.n_edit = self.findChild(QtWidgets.QLineEdit, "n_edit")
        self.p_edit = self.findChild(QtWidgets.QLineEdit, "p_edit")
        self.b_edit = self.findChild(QtWidgets.QLineEdit, "b_edit")
        self.f_edit = self.findChild(QtWidgets.QLineEdit, "f_edit")
        self.u_edit = self.findChild(QtWidgets.QLineEdit, "u_edit")
        self.d_edit = self.findChild(QtWidgets.QLineEdit, "d_edit")
        self.x_faraday_edit = self.findChild(QtWidgets.QLineEdit, "x_faraday_edit")
        self.xo_edit = self.findChild(QtWidgets.QLineEdit, "xo_edit")
        self.yo_edit = self.findChild(QtWidgets.QLineEdit, "yo_edit")
        self.v_xo_edit = self.findChild(QtWidgets.QLineEdit, "v_xo_edit")
        self.v_yo_edit = self.findChild(QtWidgets.QLineEdit, "v_yo_edit")
        self.phi_edit = self.findChild(QtWidgets.QLineEdit, "phi_edit")
        self.red_edit = self.findChild(QtWidgets.QLineEdit, "red_edit")
        self.endgeschw_edit = self.findChild(QtWidgets.QLineEdit, "endgeschw_edit")
        self.anfangsgeschw_edit = self.findChild(QtWidgets.QLineEdit, "anfangsgeschw_edit")
        self.endenenergie_edit = self.findChild(QtWidgets.QLineEdit, "endenenergie_edit")
        self.anfangsenergie_edit = self.findChild(QtWidgets.QLineEdit, "anfangsenergie_edit")
        self.endradius_edit = self.findChild(QtWidgets.QLineEdit, "endradius_edit")
        self.anfangsradius_edit = self.findChild(QtWidgets.QLineEdit, "anfangsradius_edit")
        self.bahnlange_edit = self.findChild(QtWidgets.QLineEdit, "bahnlange_edit")
        self.fzyk_edit = self._find_child_recursive(QtWidgets.QLineEdit, "fzyk_edit")

        # Find f_Slider and b_Slider (b_Slider is re-assigned here, which is acceptable)
        self.f_slider = self.findChild(QtWidgets.QSlider, "f_Slider")
        self.b_slider = self.findChild(QtWidgets.QSlider, "b_Slider")
        self.fzyk_label = self._find_child_recursive(QtWidgets.QLabel, "fzyk_label")
        self.loesch_Button = self.findChild(QtWidgets.QPushButton, "loesch_Button")
        self.export_Button = self.findChild(QtWidgets.QPushButton, "export_Button")

        if not self.fzyk_label:
            print("Error: Could not find the 'fzyk_label' in the UI file.")
            sys.exit(1)

        # Set up validators for input fields
        self._setup_validators()

    def _setup_validators(self):
        """Sets up validators for the input QLineEdit fields to ensure correct data types."""

        double_validator = QDoubleValidator()
        int_validator = QIntValidator()

        # Apply validators to relevant fields, checking for None

        if self.q_edit:
            self.q_edit.setValidator(int_validator) #Integer
        if self.b_edit:
            self.b_edit.setValidator(double_validator)
        if self.f_edit:
            self.f_edit.setValidator(double_validator)
        if self.u_edit:
            self.u_edit.setValidator(double_validator)  # Assuming this needs to be a float as well
        if self.d_edit:
            self.d_edit.setValidator(double_validator)  # Assuming this needs to be a float as well
        if self.x_faraday_edit:
            self.x_faraday_edit.setValidator(double_validator)
        if self.xo_edit:
            self.xo_edit.setValidator(double_validator)
        if self.yo_edit:
            self.yo_edit.setValidator(double_validator)
        if self.v_xo_edit:
            self.v_xo_edit.setValidator(double_validator)
        if self.v_yo_edit:
            self.v_yo_edit.setValidator(double_validator)
        if self.phi_edit:
            self.phi_edit.setValidator(double_validator)
        if self.red_edit:
            self.red_edit.setValidator(double_validator)
        if self.endgeschw_edit:
            self.endgeschw_edit.setValidator(double_validator)
        if self.anfangsgeschw_edit:
            self.anfangsgeschw_edit.setValidator(double_validator)
        if self.endenenergie_edit:
            self.endenenergie_edit.setValidator(double_validator)
        if self.anfangsenergie_edit:
            self.anfangsenergie_edit.setValidator(double_validator)
        if self.endradius_edit:
            self.endradius_edit.setValidator(double_validator)
        if self.anfangsradius_edit:
            self.anfangsradius_edit.setValidator(double_validator)
        if self.bahnlange_edit:
            self.bahnlange_edit.setValidator(double_validator)
        if self.fzyk_edit:
            self.fzyk_edit.setValidator(double_validator)

        if self.n_edit:
            self.n_edit.setValidator(int_validator) #Integer
        if self.p_edit:
            self.p_edit.setValidator(int_validator) #Integer

    def _find_child_recursive(self, class_type, object_name):
        for child in self.findChildren(class_type):
            if child.objectName() == object_name:
                return child
        return None

    def _connect_signals(self):

        # Check for None before connecting signals
        if self.q_edit:
            self.q_edit.textChanged.connect(self.update_calculations)
            self.q_edit.textEdited.connect(lambda: self._validate_input(self.q_edit, "integer"))
        if self.b_edit:
            self.b_edit.textChanged.connect(self.update_calculations)
            self.b_edit.textEdited.connect(lambda: self._validate_input(self.b_edit, "float"))
        if self.n_edit:
            self.n_edit.textChanged.connect(self.update_calculations)
            self.n_edit.textEdited.connect(lambda: self._validate_input(self.n_edit, "integer"))
        if self.p_edit:
            self.p_edit.textChanged.connect(self.update_calculations)
            self.p_edit.textEdited.connect(lambda: self._validate_input(self.p_edit, "integer"))
        if self.f_edit:
            self.f_edit.textChanged.connect(self.update_calculations)
            self.f_edit.textEdited.connect(lambda: self._validate_input(self.f_edit, "float"))
        if self.u_edit:
            self.u_edit.textChanged.connect(self.update_calculations)
            self.u_edit.textEdited.connect(lambda: self._validate_input(self.u_edit, "float"))
        if self.d_edit:
            self.d_edit.textChanged.connect(self.update_calculations)
            self.d_edit.textEdited.connect(lambda: self._validate_input(self.d_edit, "float"))
        if self.x_faraday_edit:
            self.x_faraday_edit.textChanged.connect(self.update_calculations)
            self.x_faraday_edit.textEdited.connect(lambda: self._validate_input(self.x_faraday_edit, "float"))
        if self.xo_edit:
            self.xo_edit.textChanged.connect(self.update_calculations)
            self.xo_edit.textEdited.connect(lambda: self._validate_input(self.xo_edit, "float"))
        if self.yo_edit:
            self.yo_edit.textChanged.connect(self.update_calculations)
            self.yo_edit.textEdited.connect(lambda: self._validate_input(self.yo_edit, "float"))
        if self.v_xo_edit:
            self.v_xo_edit.textChanged.connect(self.update_calculations)
            self.v_xo_edit.textEdited.connect(lambda: self._validate_input(self.v_xo_edit, "float"))
        if self.v_yo_edit:
            self.v_yo_edit.textChanged.connect(self.update_calculations)
            self.v_yo_edit.textEdited.connect(lambda: self._validate_input(self.v_yo_edit, "float"))
        if self.phi_edit:
            self.phi_edit.textChanged.connect(self.update_calculations)
            self.phi_edit.textEdited.connect(lambda: self._validate_input(self.phi_edit, "float"))
        if self.red_edit:
            self.red_edit.textChanged.connect(self.update_calculations)
            self.red_edit.textEdited.connect(lambda: self._validate_input(self.red_edit, "float"))
        if self.endgeschw_edit:
            self.endgeschw_edit.textChanged.connect(self.update_calculations)
            self.endgeschw_edit.textEdited.connect(lambda: self._validate_input(self.endgeschw_edit, "float"))
        if self.anfangsgeschw_edit:
            self.anfangsgeschw_edit.textChanged.connect(self.update_calculations)
            self.anfangsgeschw_edit.textEdited.connect(lambda: self._validate_input(self.anfangsgeschw_edit, "float"))
        if self.endenenergie_edit:
            self.endenenergie_edit.textChanged.connect(self.update_calculations)
            self.endenenergie_edit.textEdited.connect(lambda: self._validate_input(self.endenenergie_edit, "float"))
        if self.anfangsenergie_edit:
            self.anfangsenergie_edit.textChanged.connect(self.update_calculations)
            self.anfangsenergie_edit.textEdited.connect(lambda: self._validate_input(self.anfangsenergie_edit, "float"))
        if self.endradius_edit:
            self.endradius_edit.textChanged.connect(self.update_calculations)
            self.endradius_edit.textEdited.connect(lambda: self._validate_input(self.endradius_edit, "float"))
        if self.anfangsradius_edit:
            self.anfangsradius_edit.textChanged.connect(self.update_calculations)
            self.anfangsradius_edit.textEdited.connect(lambda: self._validate_input(self.anfangsradius_edit, "float"))
        if self.bahnlange_edit:
            self.bahnlange_edit.textChanged.connect(self.update_calculations)
            self.bahnlange_edit.textEdited.connect(lambda: self._validate_input(self.bahnlange_edit, "float"))
        if self.fzyk_edit:
            self.fzyk_edit.textChanged.connect(self.update_calculations)
            self.fzyk_edit.textEdited.connect(lambda: self._validate_input(self.fzyk_edit, "float"))

        self.b_slider.valueChanged.connect(self._update_b_input_from_slider)
        self.f_slider.valueChanged.connect(self._update_f_input_from_slider)
        self.b_edit.textChanged.connect(self._update_b_slider_from_input)
        self.f_edit.textChanged.connect(self.update_f_slider_from_input)

        if self.loesch_Button:
            self.loesch_Button.clicked.connect(self._loesch_button_clicked)

        if self.export_Button:
            self.export_Button.clicked.connect(self.export_to_excel)

    def _initialize_values(self):
        if self.b_edit:
            self.b_edit.setText("0.5")  # Set initial value for b_edit
            self._update_b_slider_from_input() #update the slider
        if self.f_edit:
            self.f_edit.setText("2.82")
            self.update_f_slider_from_input()

    def update_calculations(self):
        try:
            q_value = int(self.q_edit.text()) if self.q_edit and self._is_valid_int(self.q_edit.text()) else 0
            b_value = float(self.b_edit.text()) if self.b_edit and self._is_valid_float(self.b_edit.text()) else 0
            n_value = int(self.n_edit.text()) if self.n_edit and self._is_valid_int(self.n_edit.text()) else 0
            p_value = int(self.p_edit.text()) if self.p_edit and self._is_valid_int(self.p_edit.text()) else 0

            charge = 1.6e-19
            mass_proton = 1.67e-27
            mass_neutron = 1.675e-27

            particle_mass = p_value * mass_proton + n_value * mass_neutron

            if particle_mass == 0:
                f_zyc = 0  # Avoid division by zero
            else:
                f_zyc = (charge * q_value * b_value) / (2 * 3.14159 * particle_mass) if q_value and b_value else 0

            if self.fzyk_label:  # Check if fzyk_label exists
                self.fzyk_label.setText(f"{f_zyc / 1e6:.2f} MHz")
        except ValueError as e:
            print(f"Error in calculations: {e}")

    def export_to_excel(self):
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx)")

            if not file_path:
                return  # User cancelled the dialog

            wb = Workbook()
            ws = wb.active
            ws.title = "Simulation Data"

            # Add timestamp to the Excel file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ws.append(["Export Timestamp", timestamp])
            ws.append([])  # Add an empty row for better readability

            headers = ["Parameter", "Value"]
            ws.append(headers)

            # Collect data from all input fields
            data = [
                ["Charge (q)", self.q_edit.text() if self.q_edit else "-"],
                ["Magnetic Field (B)", self.b_edit.text() if self.b_edit else "-"],
                ["Neutrons (n)", self.n_edit.text() if self.n_edit else "-"],
                ["Protons (p)", self.p_edit.text() if self.p_edit else "-"],
                ["Cyclotron Frequency (f_zyc)", self.fzyk_label.text() if self.fzyk_label else "-"],
                ["Frequency (f)", self.f_edit.text() if self.f_edit else "-"],
                ["Faraday's Constant (x_faraday)", self.x_faraday_edit.text() if self.x_faraday_edit else "-"],
                ["Initial X Position (xo)", self.xo_edit.text() if self.xo_edit else "-"],
                ["Initial Y Position (yo)", self.yo_edit.text() if self.yo_edit else "-"],
                ["Initial X Velocity (v_xo)", self.v_xo_edit.text() if self.v_xo_edit else "-"],
                ["Initial Y Velocity (v_yo)", self.v_yo_edit.text() if self.v_yo_edit else "-"],
                ["U Value", self.u_edit.text() if self.u_edit else "-"],
                ["D Value", self.d_edit.text() if self.d_edit else "-"],
                ["Phi Value", self.phi_edit.text() if self.phi_edit else "-"],
                ["Red Value", self.red_edit.text() if self.red_edit else "-"],
                ["End Geschw", self.endgeschw_edit.text() if self.endgeschw_edit else "-"],
                ["Anfangs Geschw", self.anfangsgeschw_edit.text() if self.anfangsgeschw_edit else "-"],
                ["End Energie", self.endenenergie_edit.text() if self.endenenergie_edit else "-"],
                ["Anfangs Energie", self.anfangsenergie_edit.text() if self.anfangsenergie_edit else "-"],
                ["End Radius", self.endradius_edit.text() if self.endradius_edit else "-"],
                ["Anfangs Radius", self.anfangsradius_edit.text() if self.anfangsradius_edit else "-"],
                ["Bahn Lange", self.bahnlange_edit.text() if self.bahnlange_edit else "-"],
                ["Fzyk Edit", self.fzyk_edit.text() if self.fzyk_edit else "-"]
            ]

            for row in data:
                ws.append(row)

            wb.save(file_path)
            print(f"Data exported to {file_path}")
            QMessageBox.information(self, "Export Successful", f"Data exported to {file_path}")


        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            QMessageBox.critical(self, "Export Error", f"Error exporting to Excel: {e}")

    def _update_b_input_from_slider(self):
        if self.b_edit:
            self.b_edit.setText(f"{self.b_slider.value() / 100.0:.2f}")

    def _update_b_slider_from_input(self):
        if self.b_edit:  # Check if b_edit exists
            text = self.b_edit.text()
            if self._is_valid_float(text):
                try:
                    value = int(float(text) * 100)
                    # Ensure the value is within the slider's range
                    value = max(self.b_slider.minimum(), min(value, self.b_slider.maximum()))
                    self.b_slider.setValue(value)

                except ValueError:
                    pass

    def _update_f_input_from_slider(self):
        if self.f_edit:  # Check if f_edit exists
            self.f_edit.setText(f"{self.f_slider.value() / 100.0:.2f}")

    def update_f_slider_from_input(self):
        if self.f_edit:  # Check if f_edit exists
            text = self.f_edit.text()
            if self._is_valid_float(text):
                try:
                    self.f_slider.setValue(int(float(text) * 100))
                except ValueError:
                    pass

    def _loesch_button_clicked(self):
        for line_edit in self.findChildren(QtWidgets.QLineEdit):
            line_edit.clear()
            # Reset the style to default when clearing
            line_edit.setStyleSheet("")
            self.statusBar.setStyleSheet(self.status_bar_default_stylesheet)
            self.statusBar.showMessage("Ready")  # reset status bar
            self.blink_timer.stop()

    def _validate_input(self, line_edit, data_type):
        """Validates the input in a QLineEdit and provides visual feedback.
           data_type can be "integer" or "float"
        """
        if line_edit:  # Check if line_edit exists
            validator = line_edit.validator()
            text = line_edit.text()
            if validator:
                state = validator.validate(text, 0)[0]
            else:
                state = QValidator.State.Acceptable  # If no validator, accept all

            if state == QValidator.State.Acceptable:
                line_edit.setStyleSheet("")  # Reset style if valid
                self.statusBar.setStyleSheet(self.status_bar_default_stylesheet)  # Restore default style
                self.statusBar.showMessage("Ready")  # clear message
                self.blink_timer.stop()  # Stop blinking
                self.is_dimmed = False  # Reset color

            else:
                line_edit.setStyleSheet("QLineEdit { border: 2px solid red; }")
                if data_type == "integer":
                    self.statusBar.showMessage("Invalid data type entered. Please enter an integer number.")
                elif data_type == "float":
                    self.statusBar.showMessage("Invalid data type entered. Please enter a floating-point number.")
                else:
                    self.statusBar.showMessage("Invalid data entered.")
                self.start_blinking()  # Start blinking status bar when wrong data type

    def _is_valid_float(self, text):
        """Checks if the given text can be converted to a float."""
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _is_valid_int(self, text):
        """Checks if the given text can be converted to an integer."""
        try:
            int(text)
            return True
        except ValueError:
            return False

    def start_blinking(self):
        """Starts the timer that makes the status bar blink."""
        self.is_dimmed = False  # reset color
        self.blink_timer.start(500)  # Blink every 500 milliseconds

    def blink_status_bar(self):
        """Changes the status bar color to create a blinking effect."""
        dim_red = "QColor(255, 100, 100)"  # Dimmer color
        if self.is_dimmed:
            self.statusBar.setStyleSheet("background-color: lightblue;")
        else:
            self.statusBar.setStyleSheet(f"background-color: {dim_red};")
        self.is_dimmed = not self.is_dimmed


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


    #use Fusion not windows, tap key pass, put the entrys in list so that he can use that, signals and slots report,-graph widget,promot special widget