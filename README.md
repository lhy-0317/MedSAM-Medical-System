# MedSAM 智能医学影像诊断系统

这是基于 [MedSAM](https://github.com/bowang-lab/MedSAM) 的增强版图形用户界面 (GUI) 应用，提供直观、易用的医学影像交互式分割功能。

## 🚀 主要功能特点

### 多模式交互分割
1. **Box Prompt（框选模式）**
   - 在图像上绘制矩形框
   - 系统自动分割框内目标
   - 适合初步定位目标区域

2. **Point Prompt（点选模式）**
   - 左键点击添加绿色前景点
   - 右键点击添加红色背景点
   - 支持多次点击优化分割结果
   - 适合精确调整分割边界

3. **Text Prompt（文本模式）**
   - 输入器官名称（如 liver, kidney）
   - 系统自动识别并分割对应器官
   - 无需手动标注，适合快速分割已知器官

### 多格式医学影像支持
- **常规图像**：PNG、JPG、JPEG、BMP
- **医学影像**：NPY、NIfTI (.nii/.nii.gz)
- 自动处理 2D 灰度图像转换为 3 通道显示
- 支持 CT 扫描切片自动提取（NIfTI 格式）

### 实时可视化与结果保存
- 实时显示分割结果（红色半透明掩码）
- 可视化交互提示（蓝色框、彩色点）
- 支持保存分割结果为 PNG 格式

## 📋 安装说明

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- PyQt5
- CUDA 11.7+（推荐，支持 GPU 加速）

### 安装步骤

1. 创建虚拟环境并激活
   ```bash
   conda create -n medsam python=3.10 -y
   conda activate medsam
   ```

2. 安装 PyTorch（根据你的 CUDA 版本选择）
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. 克隆项目并安装依赖
   ```bash
   git clone https://github.com/bowang-lab/MedSAM
   cd MedSAM
   pip install -e .
   ```

4. 安装 GUI 依赖
   ```bash
   pip install PyQt5 SimpleITK
   ```

5. 可选：安装 transformers 以支持文本提示功能
   ```bash
   pip install transformers
   ```

## 📁 模型权重准备

下载以下预训练权重文件并放置在 `work_dir/` 目录下：

| 模式 | 权重文件 | 下载链接 |
|------|----------|----------|
| Box Prompt | medsam_vit_b.pth | [Google Drive](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) |
| Point Prompt | medsam_point_prompt_flare22.pth | [Google Drive](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) |
| Text Prompt | medsam_text_prompt_flare22.pth | [Google Drive](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) |

## 🎯 使用指南

### 启动应用

```bash
python main.py
```

### 基本操作流程

1. **加载影像**
   - 点击「加载医学影像」按钮
   - 选择支持的图像文件
   - 系统自动处理并显示图像

2. **选择分割模式**
   - 在「分割模式」面板选择交互方式
   - 系统自动加载对应模型权重

3. **执行分割**
   - **Box 模式**：按住鼠标左键拖拽绘制矩形框
   - **Point 模式**：左键点击添加前景点，右键点击添加背景点
   - **Text 模式**：在文本框输入器官名称，点击「开始文本分割」

4. **保存结果**
   - 点击「保存分割结果」按钮
   - 选择保存路径和文件名

## 📊 模式详解

### Box Prompt（框选模式）
- **适用场景**：快速分割较大的器官或病变区域
- **操作方法**：按住鼠标左键拖拽绘制蓝色矩形框
- **特点**：分割速度快，适合初步定位

### Point Prompt（点选模式）
- **适用场景**：精确分割边界复杂的目标
- **操作方法**：
  - 左键点击：添加绿色前景点
  - 右键点击：添加红色背景点
  - 每次点击后自动更新分割结果
- **特点**：精度高，适合精细调整

### Text Prompt（文本模式）
- **适用场景**：已知器官名称的快速分割
- **操作方法**：在文本框输入器官名称（英文），如 "liver"、"kidney"、"spleen"
- **特点**：无需手动标注，智能化程度高
- **注意**：需要安装 transformers 库，支持的器官名称取决于预训练数据

## 🔧 技术细节

### 核心组件

1. **TextPromptEncoder**
   - 扩展 SAM 的提示编码器，支持文本输入
   - 集成 CLIP 文本模型进行语义编码

2. **MedSAM**
   - 组合图像编码器、掩码解码器和提示编码器
   - 支持多种提示类型的灵活切换

3. **MedSAMEngine**
   - 处理模型加载和管理
   - 计算图像特征和执行分割推理

4. **MainWindow**
   - 构建用户界面
   - 处理所有用户交互事件

### 性能优化
- **GPU 加速**：自动检测 CUDA 设备
- **模型动态加载**：根据模式选择加载对应权重
- **内存管理**：使用 `torch.cuda.empty_cache()` 优化内存使用

## ❓ 常见问题

### Q: 文本分割功能不可用？
A: 请确保已安装 transformers 库：`pip install transformers`

### Q: 加载 NIfTI 文件时出错？
A: 请确保已安装 SimpleITK：`pip install SimpleITK`

### Q: 模型加载失败？
A: 请检查权重文件路径是否正确，确保文件存在于 `work_dir/` 目录下

### Q: 分割结果不准确？
A: 尝试使用 Point Prompt 模式添加更多前景/背景点进行优化

## 📄 许可证

该项目基于 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📖 参考

```
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={654},
  year={2024}
}
```

## 📞 联系方式

如有问题或建议，请通过 GitHub Issues 联系我们。

---

**MedSAM 智能医学影像诊断系统** - 让医学影像分割更简单、更精确！