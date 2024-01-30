import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ui_form import UIForm
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import median_filter
from PyQt5.QtGui import QFont, QIcon

class ImageViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Create and set up the UI form
        self.ui_form = UIForm()
        self.setCentralWidget(self.ui_form)

        # Setup connections for UI elements
        self.setup_connections()

    def setup_connections(self):
        # Connect UI elements from ui_form
        self.ui_form.choose_button.clicked.connect(self.choose_image)
        self.ui_form.choose_groundtruth_button.clicked.connect(self.choose_groundtruth_image)
        self.ui_form.evaluate_segment_button.clicked.connect(self.evaluate_segmentation)
        # self.ui_form.histogram_button.clicked.connect(self.show_hist)
        self.ui_form.histogram_eq_button.clicked.connect(self.hist_equalize)
        self.ui_form.average_button.clicked.connect(self.average_filt)
        self.ui_form.median_button.clicked.connect(self.median_filt)
        self.ui_form.kapur_button.clicked.connect(self.kapur_threshold)
        self.ui_form.otsu_button.clicked.connect(self.otsu_threshold)

        # Other attributes
        self.image_path = None

    def choose_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_path, _ = file_dialog.getOpenFileName(self, "Choose Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")

        if file_path:
            self.image_path = file_path
            self.load_image()
            self.display_image(self.image, "original")

    def choose_groundtruth_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_path, _ = file_dialog.getOpenFileName(self, "Choose Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")

        if file_path:
            self.image_path = file_path
            self.load_groundtruth_image()
            self.display_image(self.groundtruth_image, "groundtruth")

    def load_image(self):
        self.image = Image.open(self.image_path).convert("RGB")

    def load_groundtruth_image(self):
        self.groundtruth_image = Image.open(self.image_path).convert("RGB")
        
    
    def convert_pil_to_qimage(self, pil_image):
        width, height = pil_image.size
        image_data = pil_image.tobytes("raw", "RGB")
        bytes_per_line = 3 * width  # 3 bytes per pixel (RGB)
        q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format_RGB888)
        return q_image

    def display_image(self, input_image, image_type="original"):
        width, height = input_image.size
        print(f"Image size: {width} x {height}, image type: {image_type}")
        print(input_image)
        max_dimension = 300  # Set the max dimension for width or height

        # Calculate the scale factor while maintaining the aspect ratio
        scale_factor = min(max_dimension / width, max_dimension / height)
        new_width = int(scale_factor * width)
        new_height = int(scale_factor * height)

        # Resize the image with the new dimensions while maintaining aspect ratio
        resized_image = input_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to QImage and then to QPixmap
        q_image = self.convert_pil_to_qimage(resized_image)
        pixmap = QPixmap.fromImage(q_image)

        if image_type == "original":
            self.image = resized_image
            self.processed_image = resized_image
            label = self.ui_form.image_label
            self.ui_form.processing_techniques_textbox.setText("New image uploaded.")
        elif image_type == "groundtruth":
            self.groundtruth_image = resized_image
            label = self.ui_form.groundtruth_image_label
            # Update the processing techniques textbox
            current_text = self.ui_form.processing_techniques_textbox.toPlainText()
            updated_text = f"{current_text}\nNew groundtruth image uploaded."
            self.ui_form.processing_techniques_textbox.setText(updated_text)
        else:  # For processed images
            self.processed_image = resized_image
            label = self.ui_form.processed_image_label
            # Update the processing techniques textbox
            current_text = self.ui_form.processing_techniques_textbox.toPlainText()
            updated_text = f"{current_text}\n{image_type.capitalize()} applied."
            self.ui_form.processing_techniques_textbox.setText(updated_text)

        label.setPixmap(pixmap)
        label.setFixedSize(new_width, new_height)  # Ensure the label respects the image aspect ratio
        # label.setScaledContents(True)
    
    # def show_hist(self):
    #     r, g, b = self.image.split()
    #     print(type(r.histogram()))

    def calculate_psnr(self, image_1, image_2):
        if image_1.mode == 'RGB':
            image_1 = np.array(ImageOps.grayscale(image_1))
        if image_2.mode == 'RGB':
            image_2 = np.array(ImageOps.grayscale(image_2))
        mse = np.mean((image_1 - image_2)**2)

        max_pixel_value = 255
        psnr = 20*np.log10(max_pixel_value/np.sqrt(mse))

        return psnr
    
    def evaluate_segmentation(self, threshold=128):
        """
        Process the images, centrally crop them to the smallest common size, threshold to binary,
        and then calculate sensitivity, specificity, accuracy, and FPpI for image segmentation.

        Args:
            segmented_pil: PIL Image representing the segmented image.
            ground_truth_pil: PIL Image representing the ground truth image.
            threshold: Integer value to threshold the images to binary (default is 128).

        Returns:
            sensitivity, specificity, accuracy, fppi: Floats representing the calculated metrics.
        """
        segmented_pil = self.processed_image
        ground_truth_pil = self.groundtruth_image
        # Convert PIL Images to NumPy arrays and convert to grayscale
        segmented_array = np.array(segmented_pil.convert('L'))
        ground_truth_array = np.array(ground_truth_pil.convert('L'))

        print(segmented_array.shape)
        print(ground_truth_array.shape)
        print(segmented_array)
        print(ground_truth_array)

        # Determine dimensions for cropping
        min_width = min(segmented_array.shape[1], ground_truth_array.shape[1])
        min_height = min(segmented_array.shape[0], ground_truth_array.shape[0])

        # Calculate cropping coordinates
        start_x_segmented = (segmented_array.shape[1] - min_width) // 2
        start_y_segmented = (segmented_array.shape[0] - min_height) // 2

        start_x_ground_truth = (ground_truth_array.shape[1] - min_width) // 2
        start_y_ground_truth = (ground_truth_array.shape[0] - min_height) // 2

        # Crop the images centrally to the smallest common size
        segmented_cropped = segmented_array[start_y_segmented:start_y_segmented + min_height, 
                                            start_x_segmented:start_x_segmented + min_width]
        ground_truth_cropped = ground_truth_array[start_y_ground_truth:start_y_ground_truth + min_height, 
                                                start_x_ground_truth:start_x_ground_truth + min_width]
        
        print(segmented_cropped.shape)
        print(ground_truth_cropped.shape)
        print(segmented_cropped)
        print(ground_truth_cropped)

        # Threshold images to binary
        segmented_binary = (segmented_cropped > threshold).astype(int)
        ground_truth_binary = (ground_truth_cropped > threshold).astype(int)

        print(segmented_binary)
        print(ground_truth_binary)

        # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
        tp = np.sum((segmented_binary == 1) & (ground_truth_binary == 1))
        tn = np.sum((segmented_binary == 0) & (ground_truth_binary == 0))
        fp = np.sum((segmented_binary == 1) & (ground_truth_binary == 0))
        fn = np.sum((segmented_binary == 0) & (ground_truth_binary == 1))

        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        fppi = fp / np.sum(ground_truth_binary == 0) if np.sum(ground_truth_binary == 0) > 0 else 0

        # Update the textbox
        current_text = self.ui_form.processing_techniques_textbox.toPlainText()
        updated_text = f"{current_text}\nSegmentation metrics:\nsensitivity: {sensitivity}\nspecificity: {specificity}\naccuracy: {accuracy}\nfppi: {fppi}\n"
        self.ui_form.processing_techniques_textbox.setText(updated_text)

        # return sensitivity, specificity, accuracy, fppi

    def hist_equalize(self):
        input_image = self.processed_image
        if input_image.mode == 'RGB':
            input_image = ImageOps.grayscale(input_image)

        histogram = input_image.histogram()     # histogram
        # print("Hist: ", max(histogram))
        cdf = [sum(histogram[:i+1]) for i in range(len(histogram))]     # calculate CDF
        cdf_normalized = [((x - cdf[0]) / (input_image.width * input_image.height - cdf[0])) * 255 for x in cdf]        # normalize CDF
        equalized_image_data = [cdf_normalized[pixel] for pixel in input_image.getdata()]
        # print(np.array(equalized_image_data).shape)

        equalized_image = Image.new('L', input_image.size)
        equalized_image.putdata(equalized_image_data)
        equalized_image = equalized_image.convert('RGB')

        self.histogram_psnr = self.calculate_psnr(self.image, equalized_image)
        self.ui_form.histogram_psnr_label.setText("PSNR: {:.2f}".format(self.histogram_psnr))
        # self.display_image(equalized_image, "histogram")
        # After processing:
        self.processed_image = equalized_image
        self.display_image(self.processed_image, "Histogram Equalization")

    def average_filt(self):
        input_image = self.processed_image
        mask_shape = self.ui_form.average_mask_shape_combo_box.currentText()
        mask_size = int(self.ui_form.average_mask_size_line_edit.text())
        if input_image.mode == 'RGB':
            input_image = ImageOps.grayscale(input_image)

        # Kernel
        if mask_shape == 'Average':
            # filter_kernel = [[1] * mask_size for _ in range(mask_size)]
            filter_kernel = np.ones((mask_size, mask_size), dtype=float) / (mask_size * mask_size)
        elif mask_shape == 'Gaussian':
            # filter_kernel = [[1] * mask_size for _ in range(mask_size // 2)]
            y, x = np.ogrid[-mask_size//2 + 1:mask_size//2 + 1, -mask_size//2 + 1:mask_size//2 + 1]
            filter_kernel = np.exp(-(x**2 + y**2) / (2 * 1**2))
            filter_kernel /= filter_kernel.sum()
        elif mask_shape == 'Sobel (vertical)':
            filter_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif mask_shape == 'Sobel (horizontal)':
            filter_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        else:
            raise ValueError("Invalid mask_shape. Use 'square' or 'rectangle'.")

        filtered_image = convolve2d(input_image, filter_kernel, mode='same', boundary='symm')
        filtered_image = Image.fromarray(filtered_image.astype(np.uint8)).convert("RGB")

        self.average_psnr = self.calculate_psnr(self.image, filtered_image)
        self.ui_form.average_psnr_label.setText("PSNR: {:.2f}".format(self.average_psnr))
        # self.display_image(filtered_image, "average")
        # After processing:
        self.processed_image = filtered_image
        self.display_image(self.processed_image, "Average Filter")
    
    def median_filt(self):
        input_image = self.processed_image
        kernel_size = int(self.ui_form.median_mask_size_line_edit.text())
        if input_image.mode == 'RGB':
            input_image = input_image.convert('L')
        input_array = np.array(input_image)

        # Apply median filter using scipy
        filtered_array = median_filter(input_array, size=kernel_size)

        filtered_image = Image.fromarray(filtered_array).convert("RGB")

        self.median_psnr = self.calculate_psnr(self.image, filtered_image)
        self.ui_form.median_psnr_label.setText("PSNR: {:.2f}".format(self.median_psnr))
        # self.display_image(filtered_image, "median")
        # After processing:
        self.processed_image = filtered_image
        self.display_image(self.processed_image, "Median Filter")

    def kapur_threshold(self):
        input_image = self.processed_image
        input_array = np.array(input_image)
        if input_array.ndim == 3:
            image_array = np.mean(input_array, axis=-1, dtype=np.uint8)

        hist, _ = np.histogram(image_array.flatten(), bins=256, range=[0,256], density=True)
        cdf = hist.cumsum()

        entropy = np.zeros(256)

        for t in range(1, 256):
            p1 = cdf[t]
            p2 = 1 - p1

            if p1 == 0 or p2 == 0:
                entropy[t] = 0
            else:
                h1 = -((hist[:t] / p1) * np.log2(hist[:t] / p1)).sum()
                h2 = -((hist[t:] / p2) * np.log2(hist[t:] / p2)).sum()
                entropy[t] = h1 + h2

        optimal_threshold = np.argmax(entropy)

        binary_image = (input_image > optimal_threshold).astype(np.uint8) * 255
        # print(binary_image)
        filtered_image = Image.fromarray(binary_image).convert("RGB")
        # self.display_image(filtered_image, "kapur")
        # After processing:
        self.processed_image = filtered_image
        self.display_image(self.processed_image, "Kapur Segmentation")


    def otsu_threshold(self):
        input_image = self.processed_image
        input_array = np.array(input_image)
        if input_array.ndim == 3:
            image_array = np.mean(input_array, axis=-1, dtype=np.uint8)
        # Calculate histogram and normalize
        hist, bins = np.histogram(image_array.flatten(), bins=256, range=[0, 256], density=True)
        norm_hist = hist / hist.sum()

        # Initialization
        max_variance = 0
        optimal_threshold = 0

        # Iterate through possible thresholds
        for t in range(1, 256):
            w0 = norm_hist[:t].sum()
            w1 = norm_hist[t:].sum()
            mu0 = (np.arange(0, t) * norm_hist[:t]).sum() / w0 if w0 > 0 else 0
            mu1 = (np.arange(t, 256) * norm_hist[t:]).sum() / w1 if w1 > 0 else 0

            variance = w0 * w1 * (mu0 - mu1) ** 2

            if variance > max_variance:
                max_variance = variance
                optimal_threshold = t

        binary_image = (image_array > optimal_threshold).astype(np.uint8) * 255
        filtered_image = Image.fromarray(binary_image).convert("RGB")
        # self.display_image(filtered_image, "otsu")
        # After processing:
        self.processed_image = filtered_image
        self.display_image(self.processed_image, "Otsu Segmentation")

def main():
    app = QApplication(sys.argv)
    window = ImageViewerApp()
    window.setWindowTitle("Image Processor App")
    window.setFont(QFont("Arial", 12))
    window.setStyleSheet("QGroupBox:title {font-weight: bold;}")

    # Set the background color
    # window.setStyleSheet("background-color: #f0f0f0;")  # Light gray background

    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
