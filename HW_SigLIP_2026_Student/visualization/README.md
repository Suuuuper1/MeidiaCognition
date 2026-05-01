# SigLIP 图文相似度演示

基于 `google/siglip-base-patch16-224` 预训练模型，体验图文跨模态对比学习的核心结果——图文交叉相似度矩阵。

---

## 一、环境配置

**推荐使用 conda 创建独立环境：**

```bash
conda create -n siglip python=3.10 -y
conda activate siglip
pip install -r requirements.txt
```

> 若 pip 速度慢，可换用国内镜像：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 二、HuggingFace 镜像（无外网必读）

国内访问 HuggingFace 官方源受限，运行前需设置镜像：

```bash
# Linux / macOS（每次开新终端都需要重新设置，或写入 ~/.bashrc）
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 三、数据与模型准备

### 3.1 SigLIP 预训练模型（约 400 MB）

程序启动时会**自动从 HuggingFace 下载**，首次运行需要网络连接。下载后自动缓存到 `~/.cache/huggingface/`，之后无需重复下载。

**若已在其他项目下载过，可跳过此步骤。**

手动预下载（可选）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import AutoModel, AutoProcessor; \
           AutoModel.from_pretrained('google/siglip-base-patch16-224'); \
           AutoProcessor.from_pretrained('google/siglip-base-patch16-224')"
```

如需使用本地缓存路径，修改 `config.py` 中的 `MODEL_NAME`：
```python
MODEL_NAME = "./cache/siglip-base-patch16-224"
```

---

### 3.2 Flickr8K 数据集（约 1 GB）

程序启动时会**自动从 HuggingFace (`tsystems/flickr8k`) 下载**，首次运行需等待几分钟。下载后自动缓存，之后无需重复下载。

**若已缓存，可跳过此步骤。**

手动预下载（可选）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from datasets import load_dataset; load_dataset('tsystems/flickr8k', split='train')"
```

**本地数据集（可选）**：若已有标准格式的 Flickr8K 数据，将其放置到以下路径：
```
visulization/
└── Flickr8k/
    ├── Images/          # *.jpg 图像
    ├── Flickr8k.token.txt
    ├── Flickr_8k.trainImages.txt
    ├── Flickr_8k.devImages.txt
    └── Flickr_8k.testImages.txt
```
或通过环境变量指定路径：
```bash
export FLICKR8K_DIR=/path/to/your/Flickr8k
```
本地目录存在时优先使用本地数据，不再联网下载。

---

## 四、运行

```bash
cd homework_new/
export HF_ENDPOINT=https://hf-mirror.com   # 首次运行需要
python app.py
```

浏览器打开终端输出的地址（默认 `http://127.0.0.1:7860`）。

---

## 五、体验说明

界面加载完成后，点击 **「随机采样并生成热力图」** 按钮即可：

1. 从 Flickr8K 训练集随机抽取 N 张图像及其描述（N 可通过滑块调整）
2. 计算 N×N 的 Sigmoid 相似度矩阵并渲染为热力图
3. 页面下方显示本批次的图文对卡片，便于对照阅读

**观察要点：**
- 对角线（绿框）对应正样本对，颜色应明显亮于非对角线
- 多次点击「随机采样」，观察不同图文组合下矩阵的变化
- 鼠标悬停在格子上，可查看对应图像缩略图和描述文字
- 关注正/负样本对比倍数——训练充分的模型该值应 > 2×

---

## 六、目录结构

```
homework_new/
├── app.py                    # Gradio 演示主程序
├── config.py                 # 全局配置（模型名、设备、样本数等）
├── requirements.txt          # Python 依赖
├── data/
│   └── flickr8k_loader.py    # Flickr8K 数据加载（HF 或本地）
├── model/
│   └── siglip_wrapper.py     # SigLIP 模型 Wrapper
└── viz/
    └── v1_similarity_matrix.py  # 相似度矩阵热力图
```
