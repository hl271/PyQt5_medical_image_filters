# File: ui_form.py
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QLineEdit, QComboBox, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QApplication)

class UIForm(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Image Display Section
        image_layout = QGridLayout()
        self.setup_image_widgets(image_layout)
        main_layout.addLayout(image_layout)

        # Filter Options Section
        filter_layout = QGridLayout()
        self.setup_filter_widgets(filter_layout)
        main_layout.addLayout(filter_layout)

        self.setWindowTitle("Image Processing UI")

    def setup_image_widgets(self, layout):
        # Define image widgets and their positions in the grid
        self.image_label = QLabel('Enter Text:')
        self.image_hist_label = QLabel("Histogram:")
        self.image_average_label = QLabel("Average Filter:")
        self.image_median_label = QLabel("Median Filter:")
        self.image_kapur_label = QLabel("Kapur:")
        self.image_otsu_label = QLabel("Otsu:")
        layout.addWidget(self.image_label, 1, 1)
        layout.addWidget(self.image_hist_label, 1, 2)
        layout.addWidget(self.image_average_label, 2, 1)
        layout.addWidget(self.image_median_label, 2, 2)
        layout.addWidget(self.image_kapur_label, 3, 1)
        layout.addWidget(self.image_otsu_label, 3, 2)

        # Choose image button
        self.choose_button = QPushButton("Choose Image")
        layout.addWidget(self.choose_button, 4, 1, 1, 2)

        # ... add other image related widgets here with layout.addWidget(widget, row, column)

    def setup_filter_widgets(self, layout):
        # Histogram Filter Group
        histogram_group = QGroupBox("Histogram")
        histogram_layout = QVBoxLayout()
        self.histogram_button = QPushButton("Show Histogram")
        self.histogram_eq_button = QPushButton("Show Histogram Equalization")
        self.histogram_psnr_label = QLabel("PSNR:") 

        histogram_layout.addWidget(self.histogram_button)
        histogram_layout.addWidget(self.histogram_eq_button)
        histogram_layout.addWidget(self.histogram_psnr_label)
        # ... add other histogram related widgets here => DONE
        histogram_group.setLayout(histogram_layout)
        layout.addWidget(histogram_group, 0, 0)

        # Average Filter Group
        average_group = QGroupBox("Average Filter")
        average_layout = QVBoxLayout()
        self.average_button = QPushButton("Show Average Filter")
        self.average_psnr_label = QLabel("PSNR:")
        self.average_mask_size_label = QLabel("Enter Kernel Size (integer number):")
        self.average_mask_size_line_edit = QLineEdit()
        self.average_mask_shape_combo_box = QComboBox()
        self.average_mask_shape_combo_box.addItem("Average")
        self.average_mask_shape_combo_box.addItem("Gaussian")
        self.average_mask_shape_combo_box.addItem("Sobel (vertical)")
        self.average_mask_shape_combo_box.addItem("Sobel (horizontal)")

        average_layout.addWidget(self.average_button)
        average_layout.addWidget(self.average_psnr_label)
        average_layout.addWidget(self.average_mask_shape_combo_box)
        average_layout.addWidget(self.average_mask_size_label)
        average_layout.addWidget(self.average_mask_size_line_edit)
        # ... add other average filter related widgets here => DONE
        average_group.setLayout(average_layout)
        layout.addWidget(average_group, 0, 1)

        # Median Filter Group
        median_group = QGroupBox("Median Filter")
        median_layout = QVBoxLayout()
        self.median_button = QPushButton("Show Median Filter")
        self.median_psnr_label = QLabel("PSNR:")
        self.median_mask_size_label = QLabel("Enter Kernel Size (integer number):")
        self.median_mask_size_line_edit = QLineEdit()

        median_layout.addWidget(self.median_button)
        median_layout.addWidget(self.median_psnr_label)
        median_layout.addWidget(self.median_mask_size_label)
        median_layout.addWidget(self.median_mask_size_line_edit)
        # ... add other median filter related widgets here => DONE
        median_group.setLayout(median_layout)
        layout.addWidget(median_group, 0, 2)

        # Kapur Segmentation Group
        kapur_group = QGroupBox("Kapur Segmentation")
        kapur_layout = QVBoxLayout()
        self.kapur_button = QPushButton("Show Kapur's Segmentation")
        kapur_layout.addWidget(self.kapur_button)

        kapur_group.setLayout(kapur_layout)
        layout.addWidget(kapur_group, 0, 3)

        # Otsu Segmentation Group
        otsu_group = QGroupBox("Otsu Segmentation")
        otsu_layout = QVBoxLayout()
        self.otsu_button = QPushButton("Show Otsu's Segmentation")
        otsu_layout.addWidget(self.otsu_button)

        otsu_group.setLayout(otsu_layout)
        layout.addWidget(otsu_group, 0, 4)

        # ... add other filter groups similarly

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ex = UIForm()
    ex.show()
    sys.exit(app.exec_())
