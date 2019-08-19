import sys
import os

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets, QtCore
import pyqtgraph as pg

import numpy as np

from ui_main import Ui_MainWindow
from webcam import Webcam
from handpose.utils.file import to_str_digits
from handpose.utils.image import make_square

from PIL import Image

class Studio(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(Studio, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Device
        self.device_default = 0
        self.device = self.device_default

        # Webcam
        self.webcam = Webcam()

        # Image
        self.image_dir = 'outputs'
        self.image_ext = 'jpg'
        self.num_images_max_default = 10
        self.num_images_max = self.num_images_max_default
        self.num_images = 0

        self.saved_width_default = 416 # In pixel
        self.saved_height_default = 416
        self.saved_width = self.saved_width_default
        self.saved_height = self.saved_height_default

        # Filename prefix
        self.filename_prefix = 'class_memo'

        # Recording flag
        self.is_recording = False

        # Timer
        self.timer_is_on = False
        self.timer_duration = 500 # msec
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_image)

        # Plot min/max
        self.plot_min = 0.0
        self.plot_max = -1.0

        # Initialize
        self.initialize()

    def open_webcam(self):

        # Release the resource which had been used.
        if self.webcam.is_open():
            self.webcam.release()

        self.webcam.open(self.device)
        self.process_image()

        # Show message
        self.show_message('webcam is opened.')

        # Start the timer
        if not self.timer_is_on:
            self.start_timer()

    def start_timer(self):
        self.timer_is_on = True
        self.timer.start(self.timer_duration)

    def stop_timer(self):
        self.timer_is_on = False
        self.timer.stop()

    def start_recording(self):

        self.is_recording = True
        self.num_images = 0
        self.show_message('recording frames.')

    def finish_recording(self):

        self.is_recording = False
        self.show_message('recording is finished.')

    def show_message(self, msg):
        text = 'Status: ' + msg
        self.lb_status.setText(text)

    def show_num_images(self):

        text = '{}/{}'.format(self.num_images, self.num_images_max)
        self.lb_num_images.setText(text)

    def get_image_path(self, n):

        str_num = to_str_digits(n, num_digits=5) 
        filename = self.filename_prefix + '_' + str_num + '.' + self.image_ext

        path = os.path.join(self.image_dir, filename)

        return path

    def save_image(self):
        # Save the image.

        self.num_images += 1

        if self.num_images <= self.num_images_max:
            image_path = self.get_image_path(self.num_images)
            frame = self.webcam.get_frame()
            image = Image.fromarray(frame)
            size = (self.saved_width, self.saved_height)
            image = make_square(image)

            image = image.resize(size)
            image.save(image_path)

        else:
            self.num_images =  self.num_images_max
            self.finish_recording()

        # Show the number of images
        self.show_num_images()

    def process_image(self):

        if self.webcam.is_open():

            # Show frame
            frame = self.webcam.read()
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

            # Flip the image horizontally
            image_fliped = image.mirrored(True, False)

            pixmap = QPixmap.fromImage(image_fliped)

            self.lb_image.setPixmap(pixmap)

            # Record frame
            if self.is_recording:
                self.save_image()

    def initialize(self):

        # Connect the signal and slot
        self.cb_device.activated[str].connect(self.set_device)
        self.edit_num_images_max.textChanged.connect(self.set_num_images_max)
        self.edit_saved_width.textChanged.connect(self.set_saved_width)
        self.edit_saved_height.textChanged.connect(self.set_saved_height)
        self.edit_filename_prefix.textChanged.connect(self.set_filename_prefix)


        self.btn_open.clicked.connect(self.open_webcam)
        self.btn_record.clicked.connect(self.start_recording)

        # UI
        text = str(self.num_images_max)
        self.edit_num_images_max.setText(text)

        text = str(self.saved_width)
        self.edit_saved_width.setText(text)

        text = str(self.saved_height)
        self.edit_saved_height.setText(text)

        text = str(self.filename_prefix)
        self.edit_filename_prefix.setText(text)

    def set_device(self):

        value = self.cb_device.currentIndex()

        try:
            value = int(value)
        except:
            value = self.device_default

        self.device = value


    def set_num_images_max(self):

        value = self.edit_num_images_max.text()

        try:
            value = int(value)
        except:
            value = self.num_images_max_default

        self.num_images_max = value


    def set_saved_width(self):

        value = self.edit_saved_width.text()

        try:
            value = int(value)
        except:
            value = self.saved_width_default

        self.saved_width = value


    def set_saved_height(self):

        value = self.edit_saved_height.text()

        try:
            value = int(value)
        except:
            value = self.edit_saved_height_default

        self.saved_height = value

    def set_filename_prefix(self):

        value = self.edit_filename_prefix.text()
        self.filename_prefix = value

    def add_widget(self, widget):

        widget.setParent(self.central_widget)
        self.view_layout.addWidget(widget)

    def remove_widget(self, widget):

        self.view_layout.removeWidget(widget)
        widget.setParent(None)


    def refresh_view(self):

        text = 'Remianing time: {} (sec)'.format(self.num_images)
        self.lb_num_images.setText(text)



    def closeEvent(self, event):

        self.webcam.release()





if __name__ == '__main__':

    app = QApplication([])
    studio = Studio()
    studio.show()
    app.exec_()

