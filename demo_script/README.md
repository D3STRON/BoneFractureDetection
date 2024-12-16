# Bone Fracture Detection Using X-Ray Images

### Authors:
- **Anurag Sanjay Ghosh**  
- **Akash Murali**  

---

## Project Setup

### 1. Download Pre-trained Model Weights
Download the model weight files from the links below and place them in their respective folders under `modules/models`:

- **ResNet:** [Download Here](https://northeastern-my.sharepoint.com/:u:/g/personal/murali_ak_northeastern_edu/EQiETBuafthHq-wMQlSkdQUBbtCSB6KC2F4rVM9CSSfa9Q?e=qt3Aj4)  
- **VGG:** [Download Here](https://northeastern-my.sharepoint.com/:u:/g/personal/murali_ak_northeastern_edu/ESBqfFo9cUFFu1l42_KYwVsBWH1NxkI61KEl35I_79YO-Q?e=hbQkVA)  
- **YOLO:** [Download Here](https://northeastern-my.sharepoint.com/:u:/g/personal/murali_ak_northeastern_edu/EfWwlE6LxDdAtT0syOiQmJYBGm8EwX5GPIjLOjEr_Chwmg?e=CcRDyT)  

---

### 2. Environment Setup
1. Install **Conda** (if not already installed).  
2. Create a new Python environment:
For conda 
   `conda create -n demo python=3.9`
For python
   `python -m venv demo`
3. activate the newly created environment using conda activate demo
4. Install the dependencies using ' pip install -r requirements.txt' in your environment.
5. Execute the main.ipynb notebook cell by cell to run inferences on our model.
6. Additionally we have a seperate google collab notebook which needs to be used to run inference on RCNN model. 
   [RCNN google colloab link](https://colab.research.google.com/drive/1k3RO3tuwZnGK3w8KGfIZv6raazNIjTwX?usp=sharing)

# Some important links
* Dataset for localization: https://www.kaggle.com/datasets/d3stron/augmented-fracture-data
* Dataset for classification: https://www.kaggle.com/datasets/cjinny/mura-v11
* Trained ViT model: https://huggingface.co/D3STRON/bone_fracture_vit
* Trained Detr model: https://huggingface.co/D3STRON/bone-fracture-detr
* Training and Evaluation Pipelines: https://github.com/D3STRON/BoneFractureDetection
* RCNN Testing notebook: https://colab.research.google.com/drive/1k3RO3tuwZnGK3w8KGfIZv6raazNIjTwX#scrollTo=3JiFlYpE2GkL
