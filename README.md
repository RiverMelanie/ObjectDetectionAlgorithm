# ObjectDetectionAlgorithm
# 2025 具身智能机器人目标检测算法大赛

这是一个基于计算机视觉的对象检测算法项目，集成了主流的目标检测技术，可用于图像和视频中的物体识别与定位。

## 项目结构

```
ObjectDetectionAlgorithm/
├── .github/
│   └── workflows/
│       └── [CI配置文件]
├── dox/
│   └── [文档]
├── slide/
│   └── [PPT]
├── video/
│   └── [演示视频]
├── dox/
│   └── [文档]
├── src/
│   └── [算法核心代码]
├── imgs/
│   └── [测试用例]
├── .gitignore
├── README.md
├── pytest.ini
└── requirements.txt
```

## 安装配置步骤

### 克隆项目仓库

```bash
git clone https://github.com/RiverMelanie/ObjectDetectionAlgorithm.git
cd ObjectDetectionAlgorithm
```

### 创建并激活虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# Linux/Mac系统激活命令
source venv/bin/activate

# Windows系统激活命令
venv\Scripts\activate
```

### 安装依赖库

```bash
pip install -r requirements.txt
```

**环境验证命令：**

```bash
pip list | grep -E "opencv|ultralytics|numpy"
```

### 准备测试数据

```bash
mkdir -p imgs
```

### 运行代码

```bash
python src/main.py
```

## 项目说明

本项目主要实现了基于PyTorch和OpenCV的目标检测算法，支持多种检测模型的训练与推理，适用于学术研究和工业应用场景。

## 贡献方式

欢迎提交PR或Issue来完善项目，提交前请确保代码符合项目规范并通过测试。