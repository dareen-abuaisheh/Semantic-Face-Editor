# Semantic Face Editor: Text-Guided Face Attribute Editing

## Project Overview

**Semantic Face Editor** is a deep learning-based system for editing facial attributes using natural language descriptions. The project leverages state-of-the-art generative models and attribute classifiers to enable intuitive, text-guided face manipulation while preserving facial identity.

### Key Features
- **Text-Guided Editing**: Modify facial attributes using natural language descriptions
- **Identity Preservation**: Maintains facial identity while applying attribute changes
- **High-Quality Generation**: Generates realistic face images using advanced GAN architectures
- **Attribute Verification**: Ensures generated images match requested attribute modifications
- **Interactive Interface**: Web-based UI for easy access and experimentation

---

## Dataset

### CelebA Dataset
- **Source**: Large-scale face attributes dataset
- **Size**: ~200K high-resolution face images
- **Attributes**: 40 binary face attributes (gender, age, expression, etc.)
- **Resolution**: 178 × 218 pixels (typically cropped and resized)
- **Split**: Training/Validation/Test sets

**Download Instructions**:
```bash
# CelebA dataset should be downloaded from the official source
# Place raw images in: data/celeba/
# Place attribute annotations in: data/attributes/
```

---

## System Architecture

### Pipeline Overview

```
Text Input
    ↓
Text Encoder
    ↓
Attribute Vector
    ↓
Face Generator (GAN)
    ↓
Generated Face Image
    ↓
Attribute Verifier
    ↓
Final Output
```

### Three Core Components

#### 1. **Classifier Module** (`src/classifier/`)
- Facial attribute classification
- Extracts attribute vectors from face images
- Pre-trained on CelebA attributes
- Used for verification and quality assessment

#### 2. **GAN Module** (`src/gan/`)
- Generative Adversarial Network for face synthesis
- Takes attribute vectors and generates realistic faces
- Architectures: StyleGAN-based or conditional GAN
- Produces high-quality, diverse outputs

#### 3. **Verifier Module** (`src/verifier/`)
- Validates generated faces match requested attributes
- Ensures attribute consistency
- Quality assurance and feedback loop

---

## Project Structure

```
semantic-face-editor/
│
├── data/                          # Data management
│   ├── raw/                       # Original CelebA images
│   ├── processed/                 # Preprocessed data
│   ├── celeba/                    # CelebA dataset
│   └── attributes/                # Attribute annotations
│
├── models/                        # Pre-trained model weights
│   ├── classifier/                # Attribute classifier weights
│   ├── gan/                       # GAN model weights
│   └── verifier/                  # Verifier model weights
│
├── src/                           # Source code
│   ├── classifier/                # Classifier implementation
│   │   ├── __init__.py
│   │   ├── model.py              # Model architecture
│   │   ├── train.py              # Training script
│   │   └── inference.py          # Inference pipeline
│   │
│   ├── gan/                       # GAN implementation
│   │   ├── __init__.py
│   │   ├── model.py              # GAN architecture
│   │   ├── train.py              # GAN training
│   │   └── generator.py          # Generator utilities
│   │
│   ├── verifier/                  # Verifier implementation
│   │   ├── __init__.py
│   │   ├── model.py              # Verifier model
│   │   └── evaluate.py           # Evaluation metrics
│   │
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Data loading utilities
│   │   ├── preprocessing.py      # Image preprocessing
│   │   ├── visualization.py      # Visualization tools
│   │   └── metrics.py            # Evaluation metrics
│   │
│   └── app/                       # Web interface
│       ├── __init__.py
│       ├── streamlit_app.py      # Streamlit UI
│       └── gradio_app.py         # Gradio UI
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_visualization.ipynb
│
├── outputs/                       # Generated outputs
│   ├── generated/                # Generated face images
│   ├── checkpoints/              # Model checkpoints
│   └── logs/                      # Training logs
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
└── setup.sh                       # Setup script
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)
- Git

### Quick Setup

```bash
# Clone the repository
git clone <repository_url>
cd semantic-face-editor

# Run setup script (Linux/Mac)
bash setup.sh

# On Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed,celeba,attributes}
mkdir -p models/{classifier,gan,verifier}
mkdir -p outputs/{generated,checkpoints,logs}
```

---

## Training

### 1. Data Preparation
```bash
# Download CelebA dataset and place in data/celeba/
python notebooks/02_data_preprocessing.ipynb
```

### 2. Train Classifier
```bash
python src/classifier/train.py --config configs/classifier_config.yaml
```

### 3. Train GAN
```bash
python src/gan/train.py --config configs/gan_config.yaml
```

### 4. Validate with Verifier
```bash
python src/verifier/evaluate.py --model_path models/gan/best_model.pth
```

---

## Usage

### Web Interface

```bash
# Start Streamlit app
streamlit run src/app/streamlit_app.py

# Or start Gradio app
python src/app/gradio_app.py
```

### Python API

```python
from src.gan import FaceGenerator
from src.classifier import AttributeClassifier
from src.utils import preprocess_image

# Initialize models
generator = FaceGenerator(model_path='models/gan/best_model.pth')
classifier = AttributeClassifier(model_path='models/classifier/best_model.pth')

# Generate face with attributes
text_description = "a young woman with blonde hair and blue eyes"
generated_image = generator.generate_from_text(text_description)

# Verify attributes
attributes = classifier.predict(generated_image)
```

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.1.2 | Deep learning framework |
| torchvision | 0.16.2 | Computer vision utilities |
| numpy | 1.24.3 | Numerical computations |
| pandas | 2.0.3 | Data manipulation |
| opencv-python | 4.8.0 | Image processing |
| pillow | 10.0.0 | Image handling |
| matplotlib | 3.7.2 | Visualization |
| streamlit | 1.28.1 | Web interface |
| gradio | 4.4.1 | Alternative UI |
| tensorboard | 2.14.0 | Training visualization |

---

## Model Architectures

### Attribute Classifier
- **Type**: ResNet-based classifier
- **Input**: Face images (3 × 224 × 224)
- **Output**: 40 binary attribute predictions
- **Pretrained**: ImageNet weights with CelebA fine-tuning

### Generator (GAN)
- **Type**: Conditional StyleGAN/Progressive GAN
- **Input**: Attribute vector (40-dim) + noise
- **Output**: High-quality face image (256 × 256)
- **Architecture**: Multi-scale discriminator with attribute conditioning

### Verifier
- **Type**: Binary classifier
- **Purpose**: Validates attribute consistency
- **Input**: Generated image + target attributes
- **Output**: Verification score (0-1)

---

## Results & Metrics

### Evaluation Metrics
- **Identity Preservation**: LPIPS distance to source image
- **Attribute Accuracy**: Classifier prediction accuracy on generated images
- **Image Quality**: FID (Fréchet Inception Distance), IS (Inception Score)
- **Diversity**: Intra-class variation among generated samples

---

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit pull request

---

## Project Timeline (Graduation Thesis)

- **Phase 1**: Literature review & baseline implementation
- **Phase 2**: Classifier training & evaluation
- **Phase 3**: GAN architecture & training
- **Phase 4**: Verification system implementation
- **Phase 5**: Integration & UI development
- **Phase 6**: Experiments & result analysis
- **Phase 7**: Documentation & thesis writing

---

## References

- CelebA Dataset: [Liu et al., 2015]
- StyleGAN: [Karras et al., 2019]
- Conditional GANs: [Mirza & Osindero, 2014]
- Face Attribute Classification: [He et al., 2016]

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact & Support

For questions, issues, or suggestions:
- Email: your.email@university.edu
- GitHub Issues: [Create an issue]

---

**Last Updated**: 2026-05-21
