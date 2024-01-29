# File: ui_form.py
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QApplication)

class UIForm(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

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

        # Original Image Group
        origin_group = QGroupBox("Original Image")
        origin_layout = QVBoxLayout()
        self.image_label = QLabel('Placeholder:')
        origin_layout.addWidget(self.image_label)
        origin_group.setLayout(origin_layout)
        layout.addWidget(origin_group, 0, 0)

        # Histogram Group
        histogram_group = QGroupBox("Histogram Image")
        histogram_layout = QVBoxLayout()
        self.image_hist_label = QLabel("Placeholder:")
        histogram_layout.addWidget(self.image_hist_label)
        histogram_group.setLayout(histogram_layout)
        layout.addWidget(histogram_group, 0, 1)

        # Average Image Filter Group
        average_group = QGroupBox("Average Image Filter")
        average_layout = QVBoxLayout()
        self.image_average_label = QLabel("Placeholder:")
        average_layout.addWidget(self.image_average_label)
        average_group.setLayout(average_layout)
        layout.addWidget(average_group, 1, 0)

        # Median Image Filter Group
        median_group = QGroupBox("Median Image Filter")
        median_layout = QVBoxLayout()
        self.image_median_label = QLabel("Placeholder:")
        median_layout.addWidget(self.image_median_label)
        median_group.setLayout(median_layout)
        layout.addWidget(median_group, 1, 1)

        # Kapur Segmentation Group
        kapur_group = QGroupBox("Kapur Segmentation")
        kapur_layout = QVBoxLayout()
        self.image_kapur_label = QLabel("Placeholder:")
        kapur_layout.addWidget(self.image_kapur_label)
        kapur_group.setLayout(kapur_layout)
        layout.addWidget(kapur_group, 2, 0)

        # Otsu Segmentation Group
        otsu_group = QGroupBox("Otsu Segmentation")
        otsu_layout = QVBoxLayout()
        self.image_otsu_label = QLabel("Placeholder:")
        otsu_layout.addWidget(self.image_otsu_label)
        otsu_group.setLayout(otsu_layout)
        layout.addWidget(otsu_group, 2, 1)

        # Choose image button
        self.choose_button = QPushButton("Choose Image")
        layout.addWidget(self.choose_button, 3, 0, 1, 2)

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
        self.average_mask_size_label = QLabel("Choose Kernel Size :")
        self.average_mask_size_line_edit = QSpinBox()
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
        layout.addWidget(average_group, 1, 0)

        # Median Filter Group
        median_group = QGroupBox("Median Filter")
        median_layout = QVBoxLayout()
        self.median_button = QPushButton("Show Median Filter")
        self.median_psnr_label = QLabel("PSNR:")
        self.median_mask_size_label = QLabel("Choose Kernel Size :")
        self.median_mask_size_line_edit = QSpinBox()

        median_layout.addWidget(self.median_button)
        median_layout.addWidget(self.median_psnr_label)
        median_layout.addWidget(self.median_mask_size_label)
        median_layout.addWidget(self.median_mask_size_line_edit)
        # ... add other median filter related widgets here => DONE
        median_group.setLayout(median_layout)
        layout.addWidget(median_group, 2, 0)

        # Kapur Segmentation Group
        kapur_group = QGroupBox("Kapur Segmentation")
        kapur_layout = QVBoxLayout()
        self.kapur_button = QPushButton("Show Kapur's Segmentation")
        kapur_layout.addWidget(self.kapur_button)

        kapur_group.setLayout(kapur_layout)
        layout.addWidget(kapur_group, 3, 0)

        # Otsu Segmentation Group
        otsu_group = QGroupBox("Otsu Segmentation")
        otsu_layout = QVBoxLayout()
        self.otsu_button = QPushButton("Show Otsu's Segmentation")
        otsu_layout.addWidget(self.otsu_button)

        otsu_group.setLayout(otsu_layout)
        layout.addWidget(otsu_group, 4, 0)

        # ... add other filter groups similarly

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ex = UIForm()
    ex.show()
    sys.exit(app.exec_())
