import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ui_form import UIForm
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy.signal import convolve2d

class ImageViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

        # Widget for images
        self.image_label = self.ui_form.image_label
        self.image_hist_label = self.ui_form.image_hist_label
        self.image_average_label = self.ui_form.image_average_label
        self.image_median_label = self.ui_form.image_median_label
        # Choose Image
        self.choose_button = self.ui_form.choose_button
        self.choose_button.clicked.connect(self.choose_image)

        # Histogram
        self.histogram_button = self.ui_form.histogram_button
        self.histogram_button.clicked.connect(self.show_hist)
        self.histogram_eq_button = self.ui_form.histogram_eq_button
        self.histogram_eq_button.clicked.connect(self.hist_equalize)
        self.histogram_psnr_label = self.ui_form.histogram_psnr_label

        # Average Filter
        self.average_button = self.ui_form.average_button
        self.average_button.clicked.connect(self.average_filt)
        self.average_psnr_label = self.ui_form.average_psnr_label
        self.mask_size_label = self.ui_form.mask_size_label
        self.mask_size_line_edit = self.ui_form.mask_size_line_edit
        self.mask_shape_label = self.ui_form.mask_shape_label
        self.mask_shape_combo_box = self.ui_form.mask_shape_combo_box

        # Median Filter
        self.median_button = self.ui_form.median_button
        self.median_button.clicked.connect(self.median_filt)
        self.median_psnr_label = self.ui_form.median_psnr_label
        self.median_mask_size_label = self.ui_form.median_mask_size_label
        self.median_mask_size_line_edit = self.ui_form.median_mask_size_line_edit

        # Kapur's segmentation
        self.image_kapur_label = self.ui_form.image_kapur_label
        self.kapur_button = self.ui_form.kapur_button
        self.kapur_button.clicked.connect(self.kapur_threshold)

        # Otsu's segmentation
        self.image_otsu_label = self.ui_form.image_otsu_label
        self.otsu_button = self.ui_form.otsu_button
        self.otsu_button.clicked.connect(self.otsu_threshold)

        # Image path
        self.image_path = None

    def init_ui(self):
        # Create and set up the UI form
        self.ui_form = UIForm()
        self.setCentralWidget(self.ui_form)
        self.setWindowTitle('Image Processor App')

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

    def load_image(self):
        self.image = Image.open(self.image_path).convert("RGB")
    
    def convert_pil_to_qimage(self, pil_image):
        width, height = pil_image.size
        image_data = pil_image.tobytes("raw", "RGB")
        q_image = QImage(image_data, width, height, QImage.Format_RGB888)

        return q_image

    def display_image(self, input_image, image_type="original"):
        width, height = input_image.size
        resize_ratio = 300/max(width, height)
        new_width = int(resize_ratio*width)
        new_height = int(resize_ratio*height)
        print(input_image.size)
        input_image = input_image.resize((new_width,new_height))

        q_image = self.convert_pil_to_qimage(input_image)
        pixmap = QPixmap.fromImage(q_image)

        if image_type == "original":
            self.image_label.setPixmap(pixmap)
        elif image_type == "histogram":
            self.image_hist_label.setPixmap(pixmap)
        elif image_type == "average":
            self.image_average_label.setPixmap(pixmap)
        elif image_type == "median":
            self.image_median_label.setPixmap(pixmap)
        elif image_type == "kapur":
            self.image_kapur_label.setPixmap(pixmap)
        elif image_type == "otsu":
            self.image_otsu_label.setPixmap(pixmap)
    
    def show_hist(self):
        r, g, b = self.image.split()
        print(type(r.histogram()))

    def calculate_psnr(self, image_1, image_2):
        if image_1.mode == 'RGB':
            image_1 = np.array(ImageOps.grayscale(image_1))
        if image_2.mode == 'RGB':
            image_2 = np.array(ImageOps.grayscale(image_2))
        mse = np.mean((image_1 - image_2)**2)

        max_pixel_value = 255
        psnr = 20*np.log10(max_pixel_value/np.sqrt(mse))

        return psnr

    def hist_equalize(self):
        input_image = self.image
        if input_image.mode == 'RGB':
            input_image = ImageOps.grayscale(input_image)

        histogram = input_image.histogram()     # histogram
        print("Hist: ", max(histogram))
        cdf = [sum(histogram[:i+1]) for i in range(len(histogram))]     # calculate CDF
        cdf_normalized = [((x - cdf[0]) / (input_image.width * input_image.height - cdf[0])) * 255 for x in cdf]        # normalize CDF
        equalized_image_data = [cdf_normalized[pixel] for pixel in input_image.getdata()]
        print(np.array(equalized_image_data).shape)

        equalized_image = Image.new('L', input_image.size)
        equalized_image.putdata(equalized_image_data)
        equalized_image = equalized_image.convert('RGB')

        self.display_image(equalized_image, "histogram")
        self.histogram_psnr = self.calculate_psnr(self.image, equalized_image)
        self.histogram_psnr_label.setText("PSNR: {:.2f}".format(self.histogram_psnr))

    def average_filt(self):
        input_image = self.image
        mask_shape = self.mask_shape_combo_box.currentText()
        mask_size = int(self.mask_size_line_edit.text())
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

        self.display_image(filtered_image, "average")
        self.average_psnr = self.calculate_psnr(self.image, filtered_image)
        self.average_psnr_label.setText("PSNR: {:.2f}".format(self.average_psnr))
    
    def median_filt(self):
        input_image = self.image
        kernel_size = int(self.median_mask_size_line_edit.text())
        if input_image.mode == 'RGB':
            input_image = input_image.convert('L')
        input_array = np.array(input_image)

        height, width = input_array.shape

        # Define the half-size of the kernel
        half_kernel_size = kernel_size // 2

        # Create a new array for the filtered image
        filtered_array = np.zeros((height, width), dtype=np.uint8)

        for i in range(half_kernel_size, height - half_kernel_size):
            for j in range(half_kernel_size, width - half_kernel_size):
                neighborhood = input_array[i - half_kernel_size:i + half_kernel_size + 1,
                                        j - half_kernel_size:j + half_kernel_size + 1]
                filtered_array[i, j] = np.median(neighborhood)

        filtered_image = Image.fromarray(filtered_array).convert("RGB")

        self.display_image(filtered_image, "median")
        self.median_psnr = self.calculate_psnr(self.image, filtered_image)
        self.median_psnr_label.setText("PSNR: {:.2f}".format(self.median_psnr))

    def kapur_threshold(self):
        input_image = self.image
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
        print(binary_image)
        filtered_image = Image.fromarray(binary_image).convert("RGB")
        self.display_image(filtered_image, "kapur")

    def otsu_threshold(self):
        input_image = self.image
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
        self.display_image(filtered_image, "otsu")

def main():
    app = QApplication(sys.argv)
    window = ImageViewerApp()
    window.setWindowTitle("Image Viewer App")
    window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
