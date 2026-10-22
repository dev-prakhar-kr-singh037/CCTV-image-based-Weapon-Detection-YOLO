## Project Overview

This project is a Streamlit-based computer vision application designed to detect safety and security threats in CCTV images. Utilizing a custom-trained **YOLOv8** model, the system identifies and draws bounding boxes around key entities, including **persons**, **weapons** (knives, guns, rifles), and **umbrellas** (included to train the model to distinguish between weapons and false-positive items).

To build a robust training set, two separate datasets—one containing weapons and persons, and another containing weapons and umbrellas—were unified into a single pipeline using a custom tool, [yolo-data-harmoniser](https://github.com/dev-prakhar-kr-singh037/yolo-data-harmonizer).

### Key Features
- **Object Detection:** Identifies `person`, `weapon`, and `umbrella` classes in real-time or uploaded CCTV frames.
- **False-Positive Mitigation:** Trained with distractor classes (umbrellas) to reduce misclassifications in low-resolution CCTV footage.
- **Interactive UI:** Built with Streamlit for intuitive image uploads, confidence threshold adjustments, and visual inspection.
- **Data Harmonization:** Dataset pre-processing and format translation powered by the `yolo-data-harmoniser` utility.

## Dataset

The model is trained on a merged and harmonized dataset containing approximately **3,000 CCTV images** annotated with standard bounding boxes.

### Source Datasets
To improve real-world CCTV robustness and minimize false positives, two distinct sources were combined:
1. **[CCTV Weapon Detection Dataset: Rifle vs Umbrella](https://universe.roboflow.com/)** — Focuses on separating actual long-barrel weapons from everyday objects like umbrellas.
2. **[Weapon Detection CCTV v3 Dataset (v1, weapon_detection_in_cctv)](https://universe.roboflow.com/)** — Focuses on general weapon and human detection in security video feeds.

### Dataset Harmonization (`yolo-data-harmoniser`)
Because the source datasets used conflicting class IDs, mixed label schema formats, and non-uniform label structures, they were combined using **[yolo-data-harmoniser](https://github.com/your-username/yolo-data-harmoniser)**.

The harmonizer script automates:
- **Class Mapping & Alignment:** Standardizes all labels into three core unified classes:
  - `0`: `person`
  - `1`: `weapon` (knives, pistols, rifles)
  - `2`: `umbrella` (distractor class)
- **Bounding Box Normalization:** Converted varying annotation formats into normalized standard YOLO text format (`<class-id> <x_center> <y_center> <width> <height>`).
- **Data Deduplication & Split:** Merged image sets and structured them into `train`, `val`, and `test` directories with matching YAML manifests.

- ## Installation & Setup

### Prerequisites
- **Python:** 3.8+ (Python 3.10 recommended)
- **Git**

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/dev-prakhar-kr-singh037/CCTV-image-based-Weapon-Detection-YOLO.git](https://github.com/dev-prakhar-kr-singh037/CCTV-image-based-Weapon-Detection-YOLO.git)
   cd CCTV-image-based-Weapon-Detection-YOLO
Create and Activate a Virtual Environment:

On Linux/macOS:

    ```Bash
    python3 -m venv venv
    source venv/bin/activate
On Windows:

    ```Bash
    python -m venv venv
    venv\Scripts\activate
### Install Dependencies:

    ```Bash
    pip install -r requirements.txt

### Your requirements.txt should include:
Plaintext
ultralytics
streamlit
torch
opencv-python

## How to Run

### 1. Launch the Streamlit Web Application
To run the interactive web interface locally:
    ```bash
    streamlit run app.py


## Training & Testing

You can train and evaluate the model using either the provided standalone Python scripts or the interactive Jupyter Notebook included in this repository.

---

### Option A: Interactive Jupyter Notebook
To view the full end-to-end training pipeline, data visualization, and evaluation graphs interactively:
1. Open [`Weapon_Detection_Yolo.ipynb`](./Weapon_Detection_Yolo.ipynb) in **Jupyter Lab**, **VS Code**, or **Google Colab**.
2. Run the cells sequentially to visualize annotations, trigger model training, and inspect detection outputs.

---

### Option B: Standalone Python Scripts

#### 1. Training the Model (`train.py`)
To launch training locally or on a remote GPU instance:
```bash
python train.py

