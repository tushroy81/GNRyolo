import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QSlider, QMessageBox, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from ultralytics import YOLO


class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🚗 YOLO Object Detection")
        self.setStyleSheet("background-color: #1e1e2f; color: white;")
        self.setMinimumSize(1000, 800)

        self.model = YOLO("weights/best1.pt")  # Update model path
        self.device = 'cpu'
        self.result_image = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        self.title = QLabel("Vehicle Detection in Satellite Images using YOLO")
        self.title.setFont(QFont("Arial", 16, QFont.Bold))
        self.title.setStyleSheet("color: #6c63ff;")
        self.title.setAlignment(Qt.AlignCenter)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("border: 2px solid #6c63ff; background-color: #2b2d42;")
        
        # Status
        self.status = QLabel("Upload an image to begin.")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setFont(QFont("Arial", 10))
        self.status.setStyleSheet("color: #aaaaaa; margin: 10px;")

        # Confidence slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 100)
        self.slider.setValue(33)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #555;
                height: 8px;
            }
            QSlider::handle:horizontal {
                background: #6c63ff;
                border: 1px solid #777;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

        # Confidence slider label
        self.slider_label = QLabel(f"Current Threshold: {self.slider.value() / 100:.2f}")
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.slider_label.setFont(QFont("Arial", 10))
        self.slider_label.setStyleSheet("color: #ffffff; margin-bottom: 10px;")
        self.slider.valueChanged.connect(self.update_slider_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.upload_btn = self.create_button("📁 Upload Image", self.upload_image, "#ef476f", "#ff6d8f")
        self.save_btn = self.create_button("💾 Save Image", self.save_image, "#6c63ff", "#827cff")
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.save_btn)

        # Assemble layout
        layout.addWidget(self.title)
        layout.addWidget(self.image_label)
        layout.addWidget(QLabel("Confidence Threshold:"))
        layout.addWidget(self.slider)
        layout.addWidget(self.slider_label)
        layout.addWidget(self.status)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_slider_label(self):
        value = self.slider.value() / 100
        self.slider_label.setText(f"Current Threshold: {value:.2f}")

    def create_button(self, text, callback, color, hover_color):
        btn = QPushButton(text)
        btn.setFont(QFont("Arial", 11, QFont.Bold))
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)
        btn.clicked.connect(callback)
        return btn

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.run_detection(file_path)

    def run_detection(self, img_path):
        image = cv2.imread(img_path)
        conf_thres = self.slider.value() / 100

        self.status.setText("Running YOLO detection...")

        results = self.model.predict(source=image, conf=conf_thres, save=False, device=self.device)

        height, width = image.shape[:2]
        scale_factor = (height + width) / 1000  # Scaling factor based on image size
        box_thickness = max(1, int(scale_factor * 2))
        font_scale = max(0.4, scale_factor * 0.6)
        font_thickness = max(1, int(scale_factor))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{self.model.names[cls]}: {conf:.2f}'

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

                # Get size of the label text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                # cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                # cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                #             font_scale, (0, 0, 0), font_thickness)

        self.result_image = image
        self.display_image(image)
        self.status.setText("Detection complete.")
        self.save_btn.setEnabled(True)

    def display_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        scaled_pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def save_image(self):
        if self.result_image is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg)")
            if path:
                cv2.imwrite(path, self.result_image)
                QMessageBox.information(self, "Saved", "Image saved successfully.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())



