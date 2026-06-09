import sys
import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# PyQt5 导入
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
                             QWidget, QProgressBar, QCheckBox, QGroupBox)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QTimer

# 导入你设计的多尺度 Transformer 模型 (请确保已按要求修改使其返回 attention_map)
from model import ASPC_MHA


class FERSystem(QMainWindow):
    def __init__(self, weights_path):
        super().__init__()
        self.setWindowTitle("ASPC-MHA 智能面部表情监控系统 (毕设展示版)")
        self.setGeometry(100, 100, 1100, 700)  # 扩大窗口尺寸以容纳左右分栏
        self.setStyleSheet("background-color: #f0f2f5;")

        # --- 1. 初始化模型 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用 {self.device} 加载模型...")

        self.model = ASPC_MHA(num_class=7, pretrained=False)
        checkpoint = torch.load(weights_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # --- 2. 定义图像预处理 ---
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

        # --- 3. 初始化 OpenCV 模块 ---
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # --- 4. 初始化用户界面 ---
        self.initUI()

    def initUI(self):
        # 中心部件与主布局（左右分栏）
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        self.setCentralWidget(central_widget)
        central_widget.setLayout(main_layout)

        # ==================== 左侧：视觉交互区 ====================
        left_panel = QVBoxLayout()

        # 1. 主画面显示区
        self.video_label = QLabel("画面显示区域\n(请开启摄像头或上传图片)")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet(
            "border: 3px solid #ccc; background-color: black; color: white; font-size: 20px;")
        left_panel.addWidget(self.video_label)

        # 2. 控制按钮组
        btn_layout = QHBoxLayout()

        self.btn_camera = QPushButton("📷 开启摄像头")
        self.btn_camera.setFixedHeight(45)
        self.btn_camera.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        self.btn_camera.clicked.connect(self.toggle_camera)

        self.btn_upload = QPushButton("🖼️ 上传单张图片")
        self.btn_upload.setFixedHeight(45)
        self.btn_upload.setStyleSheet(
            "background-color: #0078D7; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        self.btn_upload.clicked.connect(self.upload_and_predict)

        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_upload)
        left_panel.addLayout(btn_layout)

        main_layout.addLayout(left_panel, stretch=2)

        # ==================== 右侧：数据看板区 ====================
        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignTop)

        # 1. 核心结果大屏
        result_group = QGroupBox("实时分析结果")
        result_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 16px; border: 1px solid #aaa; margin-top: 10px;} QGroupBox::title { subcontrol-origin: margin; top: -10px; left: 10px; }")
        result_layout = QVBoxLayout()

        self.lbl_main_result = QLabel("等待输入...")
        self.lbl_main_result.setAlignment(Qt.AlignCenter)
        self.lbl_main_result.setFont(QFont("Arial", 28, QFont.Bold))
        self.lbl_main_result.setStyleSheet("color: #D32F2F; margin: 10px 0px;")

        self.lbl_fps = QLabel("FPS: - | 耗时: - ms")
        self.lbl_fps.setAlignment(Qt.AlignCenter)
        self.lbl_fps.setStyleSheet("color: #666;")

        result_layout.addWidget(self.lbl_main_result)
        result_layout.addWidget(self.lbl_fps)
        result_group.setLayout(result_layout)
        right_panel.addWidget(result_group)

        # 2. 概率分布条 (Dashboard)
        prob_group = QGroupBox("表情概率分布")
        prob_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #aaa; margin-top: 10px;}")
        prob_layout = QVBoxLayout()

        self.prob_bars = {}
        for emotion in self.classes:
            row = QHBoxLayout()
            lbl = QLabel(emotion)
            lbl.setFixedWidth(70)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setStyleSheet("""
                QProgressBar { border: 1px solid #bbb; border-radius: 4px; text-align: center; height: 18px; }
                QProgressBar::chunk { background-color: #0078D7; width: 1px; }
            """)

            row.addWidget(lbl)
            row.addWidget(bar)
            prob_layout.addLayout(row)
            self.prob_bars[emotion] = bar

        prob_group.setLayout(prob_layout)
        right_panel.addWidget(prob_group)

        # 3. 注意力可视化区 (Heatmap)
        heatmap_group = QGroupBox("ASPC-MHA 注意力可视化")
        heatmap_group.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #aaa; margin-top: 10px;}")
        heatmap_layout = QVBoxLayout()
        heatmap_layout.setAlignment(Qt.AlignCenter)

        self.chk_heatmap = QCheckBox("开启特征热力图 (需消耗额外算力)")
        self.chk_heatmap.setChecked(True)

        self.heatmap_label = QLabel("暂无数据")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setFixedSize(224, 224)  # 与模型输入尺寸一致
        self.heatmap_label.setStyleSheet("border: 1px dashed #888; background-color: #ddd;")

        heatmap_layout.addWidget(self.chk_heatmap)
        heatmap_layout.addWidget(self.heatmap_label)
        heatmap_group.setLayout(heatmap_layout)
        right_panel.addWidget(heatmap_group)

        main_layout.addLayout(right_panel, stretch=1)

    # ==================== 核心业务逻辑 ====================

    def toggle_camera(self):
        if self.timer.isActive():
            # 关闭摄像头
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
            self.video_label.clear()
            self.video_label.setText("摄像头已关闭")
            self.btn_camera.setText("📷 开启摄像头")
            self.btn_camera.setStyleSheet(
                "background-color: #28a745; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        else:
            # 开启摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.lbl_main_result.setText("无法打开摄像头")
                return
            self.timer.start(30)  # 约 33fps
            self.btn_camera.setText("⏹️ 停止监控")
            self.btn_camera.setStyleSheet(
                "background-color: #DC3545; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        start_time = time.time()

        # 画面水平翻转，符合照镜子习惯
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            # 取最大的脸
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            (x, y, w, h) = faces[0]

            # 画框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 裁剪并推理
            face_img_cv = frame[y:y + h, x:x + w]
            self.process_inference(face_img_cv)

            # 计算耗时与 FPS
            infer_time = (time.time() - start_time) * 1000
            fps = 1000.0 / infer_time if infer_time > 0 else 0
            self.lbl_fps.setText(f"FPS: {fps:.1f} | 耗时: {infer_time:.1f} ms")

        # 将 OpenCV 的 BGR 图像转为 PyQt 支持的格式并显示在主屏幕
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def upload_and_predict(self):
        if self.timer.isActive():
            self.toggle_camera()  # 上传图片时先关停摄像头

        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            frame = cv2.imread(file_path)
            if frame is None:
                return

            # 显示原图在主屏幕
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(
                QPixmap.fromImage(q_img).scaled(self.video_label.width(), self.video_label.height(),
                                                Qt.KeepAspectRatio))

            # 检测脸并推理
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img_cv = frame[y:y + h, x:x + w]
                self.process_inference(face_img_cv)
            else:
                # 没找到脸就用原图硬算
                self.process_inference(frame)

    def process_inference(self, face_cv):
        try:
            # OpenCV -> PIL
            face_rgb = cv2.cvtColor(face_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(face_rgb)

            # 前向传播
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 此时，attention_map 获取到了真实的权重矩阵
                outputs, feature_map, attention_map = self.model(img_tensor)

                # 计算概率
                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy() * 100
                max_idx = np.argmax(probs)
                confidence = probs[max_idx]
                label = self.classes[max_idx]

                # 更新界面大字与进度条
                self.lbl_main_result.setText(f"{label}\n{confidence:.1f}%")
                for i, emotion in enumerate(self.classes):
                    self.prob_bars[emotion].setValue(int(probs[i]))

                # 绘制真正的热力图
                if self.chk_heatmap.isChecked() and attention_map is not None:
                    # 将 attention_map 传给绘制函数
                    self.generate_and_show_heatmap(face_rgb, attention_map)
                else:
                    self.heatmap_label.clear()
                    self.heatmap_label.setText("已关闭")

        except Exception as e:
            print(f"推理出错: {e}")

    def generate_and_show_heatmap(self, face_rgb, attention_map):
        try:
            # 1. 提取 CLS Token 对所有图像特征块的注意力权重
            # attention_map 的 shape 默认是 (Batch, Seq_Len, Seq_Len) 即 (1, 50, 50)
            # 我们只需要第 0 行 (CLS Token)，并跳过第 0 列 (CLS对自己的注意力)，取后面的 49 个空间特征块
            cls_attention = attention_map[0, 0, 1:]  # Shape: [49]

            # 2. 将一维数组还原为二维的空间特征图形状 (ResNet50 降采样 32 倍，224/32 = 7，所以是 7x7)
            spatial_size = int(np.sqrt(cls_attention.shape[0]))  # 应该是 7
            cls_attention = cls_attention.view(spatial_size, spatial_size).cpu().numpy()

            # 3. 归一化到 [0, 1] 之间，方便转换颜色
            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)

            # 4. 调整大小回 224x224 (平滑插值)
            face_resized = cv2.resize(face_rgb, (224, 224))
            heatmap_resized = cv2.resize(cls_attention, (224, 224), interpolation=cv2.INTER_CUBIC)

            # 5. 将权重映射为 JET 伪彩色 (越红表示注意力越集中，越蓝表示越不重要)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

            # 6. 将热力图与原图融合 (按 0.5 : 0.5 的透明度比例)
            face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(face_bgr, 0.5, heatmap_color, 0.5, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # 7. 转换为 PyQt 可以显示的格式
            h, w, ch = overlay_rgb.shape
            q_img = QImage(overlay_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.heatmap_label.setPixmap(QPixmap.fromImage(q_img))
        except Exception as e:
            print(f"热力图生成失败: {e}")

    def closeEvent(self, event):
        # 退出应用时确保释放摄像头
        if self.cap is not None:
            self.cap.release()
        event.accept()


if __name__ == '__main__':
    # 请确认这里的权重路径和你的本地文件一致
    WEIGHTS_PATH = './mixup_rafdb_acc0.8950.pth'

    app = QApplication(sys.argv)
    window = FERSystem(weights_path=WEIGHTS_PATH)
    window.show()
    sys.exit(app.exec_())