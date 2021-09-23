import sys
from PyQt6 import uic
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from predict_sign import predict
from description import sign_description


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.orginal_pixmap = QPixmap()
        self.sign_pixmap = QPixmap()

        self.path_2_img = str()
        self.sign = str()

        uic.loadUi('./NeuroUI.ui', self)

        self.pushButton.clicked.connect(self.btn_on_click)
        self.pushButton_2.clicked.connect(self.quit)

    def btn_on_click(self):
        self.browse()
        self.load_main_img()
        self.prediction()
        self.load_sign()
        self.show_percent()
        self.load_description()

    def quit(self):
        sys.exit(app.exec())

    def browse(self):
        self.path_2_img = QFileDialog.getOpenFileName(self, 'Open file', './test_set/')[0]

    def show_percent(self):
        percent = self.get_percent_of_success()
        self.Percent.setText(percent)

    def load_main_img(self):
        self.orginal_pixmap.load(self.path_2_img)
        pixmap = self.orginal_pixmap.scaled(851, 601)
        self.PixMapLabel.setPixmap(pixmap)
        self.prediction()

    def load_sign(self):
        path_2_sign = self.get_sign_settings()
        self.sign_pixmap.load(path_2_sign[0])
        pixmap = self.sign_pixmap.scaled(391, 341)
        self.PixMapLabel_2.setPixmap(pixmap)

    def load_description(self):
        description = self.get_sign_settings()
        self.textBrowser.setHtml(description[1])

    def prediction(self):
        self.sign = predict(self.path_2_img)
        self.get_percent_of_success()

    def get_percent_of_success(self):
        percent = self.sign[len(self.sign)-6:len(self.sign)]
        return percent

    def get_sign_settings(self):
        sign_title = self.sign[:len(self.sign)-8]
        return sign_description[sign_title]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("plastique")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())