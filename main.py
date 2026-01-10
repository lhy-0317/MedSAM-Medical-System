# -*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform, io
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QRadioButton, QButtonGroup, QLineEdit, QMessageBox, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt
from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder
from copy import deepcopy

# 尝试导入 transformers
try:
    from transformers import CLIPTokenizer, CLIPTextModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: 'transformers' library not found. Text prompt mode will not work.")

# 检查设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 1. 自定义类定义 (TextPromptEncoder & MedSAM)
# ==========================================

class TextPromptEncoder(PromptEncoder):
    def __init__(
            self,
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=1,
            activation=nn.GELU,
            model_name="openai/clip-vit-base-patch16"  # 新增参数用于指定路径
    ) -> None:
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)

        if HAS_TRANSFORMERS:
            print(f"Loading CLIP Text Model from: {model_name}")
            try:
                # 尝试加载本地或在线模型
                text_encoder = CLIPTextModel.from_pretrained(model_name)
                text_encoder.requires_grad_(False)
                self.text_encoder = text_encoder
                self.text_encoder_head = nn.Linear(512, embed_dim)
            except Exception as e:
                print(f"Error loading CLIP model: {e}")
                self.text_encoder = None
                self.text_encoder_head = None
        else:
            self.text_encoder = None
            self.text_encoder_head = None

    def forward(self, points, boxes, masks, tokens):
        bs = self._get_batch_size(points, boxes, masks, tokens)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if tokens is not None and self.text_encoder is not None:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

    def _get_batch_size(self, points, boxes, masks, tokens):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif tokens is not None:
            return tokens.shape[0]
        else:
            return 1


class MedSAM(nn.Module):
    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder,
                 freeze_image_encoder=True,
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, tokens):
        with torch.no_grad():
            image_embedding = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None, tokens=tokens,
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks


# ==========================================
# 2. 核心推理逻辑类
# ==========================================
class MedSAMEngine:
    def __init__(self):
        self.device = device
        self.model = None
        self.current_mode = None  # 'box', 'point', 'text'
        self.img_embed = None
        self.original_size = (0, 0)  # H, W
        self.tokenizer = None

        # 定义本地 CLIP 路径 (相对于 main.py 的位置)
        self.local_clip_path = "work_dir/clip_model"

        # 初始化 Tokenizer
        if HAS_TRANSFORMERS:
            if os.path.exists(self.local_clip_path):
                print(f"Loading CLIP Tokenizer from local: {self.local_clip_path}")
                try:
                    self.tokenizer = CLIPTokenizer.from_pretrained(self.local_clip_path)
                except Exception as e:
                    print(f"Failed to load local tokenizer: {e}")
            else:
                print(f"Warning: Local CLIP path '{self.local_clip_path}' does not exist. Using online default.")
                try:
                    self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
                except Exception as e:
                    print(f"Failed to init tokenizer: {e}")

        # 权重路径配置
        self.ckpt_paths = {
            'box': "work_dir/medsam_vit_b.pth",
            'point': "work_dir/medsam_point_prompt_flare22.pth",
            'text': "work_dir/medsam_text_prompt_flare22.pth"
        }

    def load_model(self, mode):
        """根据模式加载对应的模型权重"""
        if self.current_mode == mode and self.model is not None:
            return True

        ckpt_path = self.ckpt_paths.get(mode)
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found at {ckpt_path}")
            return False

        # === 显存清理 ===
        if self.model is not None:
            print("Cleaning up old model memory...")
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        # ===============

        print(f"Loading model for {mode} mode from {ckpt_path}...")

        try:
            if mode == 'text':
                # === Text 模式逻辑 ===
                if not HAS_TRANSFORMERS:
                    print("Cannot load text model: transformers library missing.")
                    return False

                # 1. 使用 box 权重初始化基础 SAM 结构
                if not os.path.exists(self.ckpt_paths['box']):
                    print("Error: Box checkpoint is required for initialization but not found.")
                    return False

                sam_model = sam_model_registry["vit_b"](checkpoint=self.ckpt_paths['box'])

                # 2. 构建自定义 TextPromptEncoder
                clip_path = self.local_clip_path if os.path.exists(
                    self.local_clip_path) else "openai/clip-vit-base-patch16"

                text_prompt_encoder = TextPromptEncoder(
                    embed_dim=256,
                    image_embedding_size=(64, 64),
                    input_image_size=(1024, 1024),
                    mask_in_chans=1,
                    activation=nn.GELU,
                    model_name=clip_path
                )

                # 3. 组装 MedSAM
                medsam_model = MedSAM(
                    image_encoder=sam_model.image_encoder,
                    mask_decoder=sam_model.mask_decoder,
                    prompt_encoder=text_prompt_encoder
                )

                # 4. 加载 Text 专用权重 (覆盖)
                print(f"Overwriting with text checkpoint: {ckpt_path} ...")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # === 【关键修改】过滤掉不兼容的 position_ids ===
                # 原因：新版 transformers/torch 可能不再注册 position_ids，但旧权重文件里有
                keys_to_ignore = [
                    "prompt_encoder.text_encoder.text_model.embeddings.position_ids"
                ]
                for k in keys_to_ignore:
                    if k in state_dict:
                        print(f"Warning: Removing incompatible key from checkpoint: {k}")
                        del state_dict[k]
                # ==============================================

                # 使用 strict=False 进一步防止其他小版本差异导致奔溃
                # 但主要问题应由上面的 del 解决了
                msg = medsam_model.load_state_dict(state_dict, strict=False)
                print(f"Text model load result: {msg}")
                self.model = medsam_model

            else:
                # === Box / Point 模式逻辑 ===
                try:
                    self.model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
                except Exception as e:
                    print(f"Standard load failed ({e}), trying dict load...")
                    # 备用加载方案
                    sam_model = sam_model_registry["vit_b"](checkpoint=self.ckpt_paths['box'])
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    if 'model' in checkpoint:
                        sam_model.load_state_dict(checkpoint['model'])
                    else:
                        sam_model.load_state_dict(checkpoint)
                    self.model = sam_model

            self.model.to(self.device)
            self.model.eval()
            self.current_mode = mode
            self.img_embed = None
            print("Model loaded successfully.")
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to load model: {e}")
            return False

    def set_image(self, img_3c):
        """预处理图像并计算Embedding"""
        if self.model is None: return

        self.original_size = img_3c.shape[:2]
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            self.img_embed = self.model.image_encoder(img_1024_tensor)
        print("Image embedding computed.")

    @torch.no_grad()
    def predict_box(self, box):
        H, W = self.original_size
        box_1024 = box / np.array([W, H, W, H]) * 1024
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
        if len(box_torch.shape) == 1:
            box_torch = box_torch[None, :]

        args = {'points': None, 'boxes': box_torch, 'masks': None}
        if self.current_mode == 'text':
            args['tokens'] = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(**args)
        return self._decode_mask(sparse_embeddings, dense_embeddings)

    @torch.no_grad()
    def predict_point(self, points, labels):
        H, W = self.original_size
        points_1024 = points / np.array([W, H]) * 1024
        point_coords = torch.as_tensor(points_1024, dtype=torch.float, device=self.device).unsqueeze(0)
        point_labels = torch.as_tensor(labels, dtype=torch.int, device=self.device).unsqueeze(0)

        args = {'points': (point_coords, point_labels), 'boxes': None, 'masks': None}
        if self.current_mode == 'text':
            args['tokens'] = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(**args)
        return self._decode_mask(sparse_embeddings, dense_embeddings)

    @torch.no_grad()
    def predict_text(self, text_prompt):
        if self.current_mode != 'text' or self.tokenizer is None:
            print("Text mode not active or tokenizer missing.")
            return None

        print(f"Text prompt: {text_prompt}")
        # Tokenize
        text_inputs = self.tokenizer(
            [text_prompt],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tokens = text_inputs.input_ids.to(self.device)

        # Encode & Decode
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None, boxes=None, masks=None, tokens=tokens
        )
        return self._decode_mask(sparse_embeddings, dense_embeddings)

    def _decode_mask(self, sparse, dense):
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.img_embed,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        low_res_pred = torch.sigmoid(low_res_logits)
        H, W = self.original_size
        low_res_pred = F.interpolate(
            low_res_pred, size=(H, W), mode="bilinear", align_corners=False
        )
        pred_mask = low_res_pred.squeeze().cpu().numpy()
        return (pred_mask > 0.5).astype(np.uint8)


# ==========================================
# 3. GUI 界面类
# ==========================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MedSAM 智能医学影像诊断系统 (AI课设演示)")
        self.resize(1200, 800)

        self.engine = MedSAMEngine()
        self.img_path = None
        self.img_3c = None
        self.mask = None

        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.prompt_points = []
        self.prompt_labels = []

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.image_label = QLabel("请加载图像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(600, 600)
        self.image_label.setMouseTracking(True)
        main_layout.addWidget(self.image_label, stretch=2)

        control_panel = QVBoxLayout()
        grp_file = QGroupBox("文件操作")
        vbox_file = QVBoxLayout()
        btn_load = QPushButton("加载医学影像")
        btn_load.clicked.connect(self.load_image)
        btn_save = QPushButton("保存分割结果")
        btn_save.clicked.connect(self.save_result)
        vbox_file.addWidget(btn_load)
        vbox_file.addWidget(btn_save)
        grp_file.setLayout(vbox_file)
        control_panel.addWidget(grp_file)

        grp_mode = QGroupBox("分割模式 (交互方式)")
        vbox_mode = QVBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.rb_box = QRadioButton("Box Prompt (框选)")
        self.rb_box.setChecked(True)
        self.rb_point = QRadioButton("Point Prompt (点击)")
        self.rb_text = QRadioButton("Text Prompt (文本)")
        self.mode_group.addButton(self.rb_box, 0)
        self.mode_group.addButton(self.rb_point, 1)
        self.mode_group.addButton(self.rb_text, 2)
        self.mode_group.buttonClicked.connect(self.change_mode)
        vbox_mode.addWidget(self.rb_box)
        vbox_mode.addWidget(self.rb_point)
        vbox_mode.addWidget(self.rb_text)
        grp_mode.setLayout(vbox_mode)
        control_panel.addWidget(grp_mode)

        grp_text = QGroupBox("文本提示")
        vbox_text = QVBoxLayout()
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("输入器官名称 (如 liver, kidney)")
        self.btn_text_seg = QPushButton("开始文本分割")
        self.btn_text_seg.clicked.connect(self.run_text_seg)
        self.input_text.setEnabled(False)
        self.btn_text_seg.setEnabled(False)
        vbox_text.addWidget(self.input_text)
        vbox_text.addWidget(self.btn_text_seg)
        grp_text.setLayout(vbox_text)
        control_panel.addWidget(grp_text)

        self.info_label = QLabel("就绪")
        self.info_label.setStyleSheet("color: blue; font-weight: bold;")
        control_panel.addWidget(self.info_label)

        btn_clear = QPushButton("清除选区/重置")
        btn_clear.clicked.connect(self.reset_interaction)
        control_panel.addWidget(btn_clear)
        control_panel.addStretch()
        main_layout.addLayout(control_panel, stretch=1)
        self.setLayout(main_layout)
        self.change_mode(self.rb_box)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", ".",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.npy *.nii *.nii.gz)")
        if not file_path: return
        try:
            self.info_label.setText(f"正在加载: {os.path.basename(file_path)}...")
            QApplication.processEvents()

            if file_path.endswith('.npy'):
                img_np = np.load(file_path)
                # 简单归一化
                if img_np.max() > img_np.min():
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0
                img_np = img_np.astype(np.uint8)
                if len(img_np.shape) == 2:
                    self.img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
                else:
                    self.img_3c = img_np
            # 新增支持 NIfTI 格式 (配合 .nii)
            elif file_path.endswith(('.nii', '.nii.gz')):
                import SimpleITK as sitk
                itk_img = sitk.ReadImage(file_path)
                img_arr = sitk.GetArrayFromImage(itk_img)
                # 默认取中间一层作为演示
                slice_idx = img_arr.shape[0] // 2
                img_2d = img_arr[slice_idx]
                # 归一化
                if img_2d.max() > img_2d.min():
                    img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min()) * 255.0
                img_2d = img_2d.astype(np.uint8)
                self.img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_np = io.imread(file_path)
                if len(img_np.shape) == 2:
                    self.img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
                else:
                    self.img_3c = img_np[:, :, :3]

            self.img_path = file_path
            self.mask = np.zeros(self.img_3c.shape[:2], dtype=np.uint8)
            self.reset_interaction()
            self.info_label.setText("正在计算图像特征 (Embedding)...")
            QApplication.processEvents()
            self.engine.set_image(self.img_3c)
            self.info_label.setText(f"加载成功: {os.path.basename(file_path)}")
            self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法加载图像: {str(e)}")

    def change_mode(self, btn):
        mode_map = {self.rb_box: 'box', self.rb_point: 'point', self.rb_text: 'text'}
        mode = mode_map.get(btn)
        self.input_text.setEnabled(mode == 'text')
        self.btn_text_seg.setEnabled(mode == 'text')
        self.reset_interaction()
        self.info_label.setText(f"正在切换模型到: {mode}...")
        QApplication.processEvents()

        # 增加一点延时让UI刷新
        time.sleep(0.1)

        success = self.engine.load_model(mode)
        if success:
            self.info_label.setText(f"当前模式: {mode}")
            if self.img_3c is not None:
                self.info_label.setText("正在重新计算特征...")
                QApplication.processEvents()
                self.engine.set_image(self.img_3c)
                self.info_label.setText(f"当前模式: {mode} (就绪)")
        else:
            self.info_label.setText("模型加载失败，请检查权重文件路径")

    def reset_interaction(self):
        self.start_point = None
        self.end_point = None
        self.prompt_points = []
        self.prompt_labels = []
        self.mask = None
        self.update_display()

    def run_text_seg(self):
        text = self.input_text.text().strip()
        if not text: return
        self.info_label.setText(f"正在进行文本分割: {text}...")
        QApplication.processEvents()
        res = self.engine.predict_text(text)
        if res is None:
            QMessageBox.information(self, "提示", "Text Prompt 失败或未配置环境。")
        else:
            self.mask = res
            self.update_display()
            self.info_label.setText(f"文本分割完成: {text}")

    def mousePressEvent(self, event):
        if self.img_3c is None: return
        local_pos = self.image_label.mapFrom(self, event.pos())
        x, y = local_pos.x(), local_pos.y()
        if self.image_label.pixmap() is None: return
        pix_w = self.image_label.pixmap().width()
        pix_h = self.image_label.pixmap().height()
        img_h, img_w = self.img_3c.shape[:2]
        real_x = int(x / pix_w * img_w)
        real_y = int(y / pix_h * img_h)
        real_x = max(0, min(real_x, img_w - 1))
        real_y = max(0, min(real_y, img_h - 1))

        if self.rb_box.isChecked():
            self.drawing = True
            self.start_point = (real_x, real_y)
            self.end_point = (real_x, real_y)
        elif self.rb_point.isChecked():
            label = 1 if event.buttons() == Qt.LeftButton else 0
            self.prompt_points.append([real_x, real_y])
            self.prompt_labels.append(label)
            self.run_point_inference()
        self.update_display()

    def mouseMoveEvent(self, event):
        if self.rb_box.isChecked() and self.drawing:
            local_pos = self.image_label.mapFrom(self, event.pos())
            x, y = local_pos.x(), local_pos.y()
            pix_w = self.image_label.pixmap().width()
            pix_h = self.image_label.pixmap().height()
            img_h, img_w = self.img_3c.shape[:2]
            real_x = int(x / pix_w * img_w)
            real_y = int(y / pix_h * img_h)
            self.end_point = (real_x, real_y)
            self.update_display()

    def mouseReleaseEvent(self, event):
        if self.rb_box.isChecked() and self.drawing:
            self.drawing = False
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                self.info_label.setText("正在推理 (Box)...")
                QApplication.processEvents()
                self.mask = self.engine.predict_box(box)
                self.update_display()
                self.info_label.setText("推理完成")

    def run_point_inference(self):
        if not self.prompt_points: return
        self.info_label.setText(f"正在推理 (Points: {len(self.prompt_points)})...")
        QApplication.processEvents()
        points = np.array(self.prompt_points)
        labels = np.array(self.prompt_labels)
        self.mask = self.engine.predict_point(points, labels)
        self.update_display()
        self.info_label.setText("推理完成")

    def update_display(self):
        if self.img_3c is None: return
        h, w, ch = self.img_3c.shape
        bytes_per_line = ch * w
        q_img = QImage(self.img_3c.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        if self.mask is not None:
            mask_rgb = np.zeros((h, w, 4), dtype=np.uint8)
            mask_rgb[self.mask > 0] = [255, 0, 0, 100]
            q_mask = QImage(mask_rgb.data, w, h, 4 * w, QImage.Format_RGBA8888)
            painter = QPainter(pixmap)
            painter.drawImage(0, 0, q_mask)
            painter.end()

        painter = QPainter(pixmap)
        pen_box = QPen(QColor("blue"), 3)
        pen_point_pos = QPen(QColor("green"), 10)
        pen_point_neg = QPen(QColor("red"), 10)

        if self.rb_box.isChecked() and self.start_point and self.end_point:
            painter.setPen(pen_box)
            painter.setBrush(Qt.NoBrush)
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            painter.drawRect(min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2))

        if self.rb_point.isChecked():
            for pt, lbl in zip(self.prompt_points, self.prompt_labels):
                painter.setPen(pen_point_pos if lbl == 1 else pen_point_neg)
                painter.drawPoint(pt[0], pt[1])

        painter.end()
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_result(self):
        if self.mask is None: return
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "result.png", "PNG Files (*.png)")
        if path:
            io.imsave(path, self.mask * 255)
            QMessageBox.information(self, "成功", f"掩码已保存至 {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())