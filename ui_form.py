# File: ui_form.py
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox,
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

        # Filter and Segmentation Options Section
        process_layout = QHBoxLayout()
        self.setup_filter_segmentation_widgets(process_layout)
        main_layout.addLayout(process_layout)


    def setup_image_widgets(self, layout):
        # Original Image Group
        origin_group = QGroupBox("Original Image")
        origin_layout = QVBoxLayout()
        self.image_label = QLabel('Placeholder:')
        origin_layout.addWidget(self.image_label)
        origin_group.setLayout(origin_layout)
        layout.addWidget(origin_group, 0, 0)

        # Processed Image Group
        groundtruth_group = QGroupBox("Groundtruth Mask")
        groundtruth_layout = QVBoxLayout()
        self.groundtruth_image_label = QLabel("Placeholder:")
        groundtruth_layout.addWidget(self.groundtruth_image_label)
        groundtruth_group.setLayout(groundtruth_layout)
        layout.addWidget(groundtruth_group, 0, 1)

        # Processed Image Group
        processed_group = QGroupBox("Processed Image")
        processed_layout = QVBoxLayout()
        self.processed_image_label = QLabel("Placeholder:")
        processed_layout.addWidget(self.processed_image_label)
        processed_group.setLayout(processed_layout)
        layout.addWidget(processed_group, 0, 2)

        # Choose original image button
        self.choose_button = QPushButton("Choose Original Image")
        layout.addWidget(self.choose_button, 1, 0)

        # Choose groundtruth mask button
        self.choose_groundtruth_button = QPushButton("Choose Groundtruth Mask")
        layout.addWidget(self.choose_groundtruth_button, 1, 1)

        # Calculate segmentation accuracy button
        self.evaluate_segment_button = QPushButton("Evaluate Segmentation")
        layout.addWidget(self.evaluate_segment_button, 1, 2)

        # Processing Techniques Textbox
        self.processing_techniques_textbox = QTextEdit()
        self.processing_techniques_textbox.setReadOnly(True)
        layout.addWidget(self.processing_techniques_textbox, 2, 0, 1, 3)

    def setup_filter_segmentation_widgets(self, layout):
        # Filter Group
        filter_group = QGroupBox("Filter")
        filter_layout = QGridLayout()
        self.setup_filter_widgets(filter_layout)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Segmentation Group
        segmentation_group = QGroupBox("Segmentation")
        segmentation_layout = QGridLayout()
        self.setup_segmentation_widgets(segmentation_layout)
        segmentation_group.setLayout(segmentation_layout)
        layout.addWidget(segmentation_group)
    def setup_filter_widgets(self, layout):
        # Histogram Filter Group
        histogram_group = QGroupBox("Histogram")
        histogram_layout = QVBoxLayout()
        # self.histogram_button = QPushButton("Show Histogram")
        self.histogram_eq_button = QPushButton("Show Histogram Equalization")
        self.histogram_psnr_label = QLabel("PSNR:") 

        # histogram_layout.addWidget(self.histogram_button)
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
        self.average_mask_size_line_edit.setRange(1, 10)
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
        self.median_mask_size_line_edit.setRange(1, 10)

        median_layout.addWidget(self.median_button)
        median_layout.addWidget(self.median_psnr_label)
        median_layout.addWidget(self.median_mask_size_label)
        median_layout.addWidget(self.median_mask_size_line_edit)
        # ... add other median filter related widgets here => DONE
        median_group.setLayout(median_layout)
        layout.addWidget(median_group, 2, 0)

    def setup_segmentation_widgets(self, layout):
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

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    ex = UIForm()
    ex.show()
    sys.exit(app.exec_())
