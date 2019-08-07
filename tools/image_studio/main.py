import sys
import types

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

import numpy as np


from ui_main import Ui_MainWindow
from webcam import Webcam


class Studio(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(Studio, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Device
        self.device_default = 0
        self.device = self.device_default

        # Webcam
        self.webcam = Webcam()

        self.image_width = 800
        self.image_height = 600

        self.view_width = 800
        self.view_height = 600

        self.exported_width_default = 800 # In pixel
        self.exported_height_default = 600
        self.exported_width = self.exported_width_default
        self.exported_height = self.exported_height_default

        self.recording_time_default = 10 # In seconds
        self.recording_time = self.recording_time_default
        self.remaining_time = 0  # In seconds


        # Timer
        self.timer = QTimer(self)
        self.timer.start(500)
        self.timer.timeout.connect(self.refresh_view)

        # Plot min/max
        self.plot_min = 0.0
        self.plot_max = -1.0

        # Set the initial values for UI
        self.set_ui_values()

        # Connect the signal and slot

        self.cb_device.activated[str].connect(self.set_device)
        self.edit_recording_time.textChanged.connect(self.set_recording_time)
        self.edit_exported_width.textChanged.connect(self.set_exported_width)
        self.edit_exported_height.textChanged.connect(self.set_exported_height)


        self.btn_play.clicked.connect(self.play_webcam)


    def play_webcam(self):

        if self.webcam.is_open():
            frame = self.webcam.read()

            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.lb_image.setPixmap(pixmap)

    def set_ui_values(self):

        text = str(self.recording_time)
        self.edit_recording_time.setText(text)

        text = str(self.exported_width)
        self.edit_exported_width.setText(text)

        text = str(self.exported_height)
        self.edit_exported_height.setText(text)

    def set_device(self):

        value = self.cb_device.currentIndex()

        try:
            value = int(value)
        except:
            value = self.device_default

        self.device = value

        # Set the device of webcam
        self.webcam.set_device(self.device)

    def set_recording_time(self):

        value = self.edit_recording_time.text()

        try:
            value = int(value)
        except:
            value = self.recording_time_default

        self.recording_time = value


    def set_exported_width(self):

        value = self.edit_exported_width.text()

        try:
            value = int(value)
        except:
            value = self.exported_width_default

        self.exported_width = value


    def set_exported_height(self):

        value = self.edit_exported_height.text()

        try:
            value = int(value)
        except:
            value = self.edit_exported_height_default

        self.exported_height = value

    def add_widget(self, widget):

        widget.setParent(self.central_widget)
        self.view_layout.addWidget(widget)

    def remove_widget(self, widget):

        self.view_layout.removeWidget(widget)
        widget.setParent(None)


    def refresh_view(self):

        text = 'Remianing time: {} (sec)'.format(self.remaining_time)
        self.lb_remaining_time.setText(text)


if __name__ == '__main__':

    app = QApplication([])
    studio = Studio()
    studio.show()
    app.exec_()

