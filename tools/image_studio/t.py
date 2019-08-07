
from PyQt5 import QtGui, QtCore, QtWidgets

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Window(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        layout = QHBoxLayout(self)
        self.button = QToolButton(self)
        self.button.setPopupMode(QToolButton.MenuButtonPopup)
        self.button.setMenu(QMenu(self.button))
        self.textBox = QTextBrowser(self)
        action = QWidgetAction(self.button)
        action.setDefaultWidget(self.textBox)
        self.button.menu().addAction(action)
        layout.addWidget(self.button)

if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.resize(100, 60)
    window.show()
    sys.exit(app.exec_())
