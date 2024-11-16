
# LLaVA-Mob: Efficient and Domain-Aligned Learners for Smartphone GUI Automation

**LLaVA-Mob** is a lightweight and efficient multimodal large language model (MLLM) agent tailored for smartphone graphical user interface (GUI) automation. It leverages optimized architecture, feature alignment, and instruction compression to achieve state-of-the-art performance on GUI automation tasks with minimal resource requirements.

---

## 🌟 **Key Features**
- **Lightweight Design**: Optimized for deployment on smartphones with reduced parameter size and inference costs.
- **Advanced Multimodal Alignment**: Enhanced feature alignment using synthetic datasets for precise collaboration between visual and text modules.
- **Instruction Compression**: Simplifies task instructions to improve speed and accuracy.
- **State-of-the-Art Performance**: Achieves top results on benchmarks like AITW and META-GUI.

---

## 🚀 **Installation**

### Prerequisites
1. Python 3.8+
2. PyTorch 1.10+
3. CUDA 11.3+ for GPU acceleration

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/LLaVA-Mob.git
   cd LLaVA-Mob
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained models and datasets:
   - [AITW Dataset](https://link-to-dataset.com)
   - [Spice Dataset](https://link-to-dataset.com)
   - [Pretrained Model](https://link-to-model.com)

   Place them in the `data/` and `models/` directories.

---

## 🛠 **Usage**

### Training
To train the model on the Spice dataset for feature alignment:
```bash
python train.py --data_dir ./data/spice --model_dir ./models/ --epochs 3
```

To fine-tune on AITW:
```bash
python train.py --data_dir ./data/aitw --model_dir ./models/ --epochs 3 --finetune
```

### Evaluation
Evaluate the model on AITW benchmark:
```bash
python evaluate.py --data_dir ./data/aitw --model_dir ./models/
```

### Inference
Run inference on a sample GUI task:
```bash
python inference.py --image sample_gui.png --output results.json
```

---

## 📊 **Benchmarks**

| Dataset       | Parameters | Overall Accuracy (%) | General | Single  |
|---------------|------------|-----------------------|---------|---------|
| **LLaVA-Mob** | 1B         | **77.73**            | 71.61   | 87.15   |

LLaVA-Mob consistently outperforms other methods while maintaining a smaller parameter size and faster inference.

---

## 📚 **Architecture**

The model consists of:
- **Text Module**: A lightweight LLaMA-3.2-1B model for efficient instruction parsing.
- **Vision Encoder**: SeeClick ViT-based encoder pre-trained on GUI-specific data.
- **Feature Alignment**: A two-layer linear projection for bridging visual and text representations.

---

## 📦 **Datasets**
- **AITW**: A large-scale benchmark for smartphone GUI automation with 1M samples.
- **AMEX**: High-resolution screenshots and element functionalities for GUI training.
- **Spice**: Synthetic dataset with detailed descriptions and precise element locations.

---

## 📝 **Citation**

If you use LLaVA-Mob in your research, please cite:

```bibtex
@article{LLaVA-Mob2024,
  title={LLaVA-Mob: Efficient and Domain-Aligned Learners for Smartphone GUI Automation},
  author={Biao Wu, Meng Fang, Zhiwei Zhang, Ling Chen},
  year={2024},
  journal={arXiv preprint},
  url={https://arxiv.org/abs/XXXX.XXXXX},
}
```

---

## 🔧 **Future Work**
- Integration with reinforcement learning for real-world deployment.
- Extending capabilities to more diverse mobile applications.

---

## 🤝 **Contributing**
Contributions are welcome! Please submit issues or pull requests to improve the project.

---

## 📜 **License**
This project is licensed under the [MIT License](LICENSE).
