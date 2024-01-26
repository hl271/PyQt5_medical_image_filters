import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_form import UIForm

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Create and set up the UI form
        self.ui_form = UIForm()
        self.setCentralWidget(self.ui_form)

        # Set window properties
        self.setWindowTitle('Simple UI Form')
        self.setGeometry(0, 0, 1800, 900)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
