
# [ AAAI2025 ] Image Regeneration: Evaluating Text-to-Image Model via Generating Identical Image with Multimodal Large Language Models
### [[Paper]](https://arxiv.org/abs/2411.09449)

## 🔥 News 

- 2025-02-28: We have release the code for Image Regeneration, supporting Qwen-vl & GPT



## To Do List
- [x] Code
- [x] Evalref-100 Dataset (Content Diverse)
- [ ] Project Page
- [ ] More results via different MLLMs





![comparch4](https://github.com/user-attachments/assets/52a39734-3207-4c0e-80e1-e1a8b7b14eb5)


# Overview

This repository introduces a novel **Image-to-Image Evaluation Framework** for Text-to-Image (T2I) models, addressing critical limitations of traditional text-image alignment methods. Our approach leverages **reference images** as input anchors, enabling a closed-loop "generate → compare → refine" pipeline. Here’s why this method outperforms conventional text-image evaluation:

---

## **Key Advantages**

### 1. **Bridging the Modality Gap**
- **Traditional Methods**: Rely on text-image similarity in embedding spaces (e.g., CLIP Score), often failing to capture fine-grained mismatches (e.g., object counts, spatial relations).
- **Our Framework**: Uses **image-to-image alignment**, reducing ambiguity by directly comparing generated outputs with reference visuals.

### 2. **Granular Feedback & Iterative Refinement**
- **Traditional Methods**: Provide static, one-time scoring.
- **Our Method**: Enables **multi-round refinement**, where multimodal feedback identifies errors (e.g., missing details) and iteratively improves prompts, mirroring human creative workflows.

### 3. **Holistic & Explainable Metrics**
- Combines **semantic fidelity** (CLIP), **structural consistency** (SSIM/DINOv2), and **creative adaptation** (MLLMs/human scoring), avoiding overreliance on single metrics.
- Visual comparisons (reference vs. generated images) offer **intuitive, explainable results**.

### 4. **Robustness to Text Ambiguity**
- **Text Prompts**: Often under-specify visual details (e.g., "a cozy room").
- **Reference Images**: Provide **unambiguous ground truth**, enabling precise evaluation of layout, style, and compositional accuracy.

---

## **Why This Framework?**

- **Benchmarking**: Evaluates T2I models’ **dynamic optimization capability**.
- **Fine-Grained Alignment**: Tests complex scenes (multi-object interactions, abstract concepts).
- **Interactive Tools**: Ideal for developing AI art tools with **human-in-the-loop refinement**.

---

## **Conclusion**
By shifting from *text-to-image* to **image-driven evaluation**, this framework sets a new standard for rigor and practicality in generative model assessment. It bridges the gap between human creativity and machine-generated art, enabling more reliable and interpretable T2I model evaluation.



# Getting Started

## 1. Environment Setup
Create a conda environment and install dependencies:
```bash
conda create -n ImageRepainter python=3.10.14
conda activate ImageRepainter
pip install -r requirement.txt
```

## 2. Model Download & File Structure

### Required Models

Download pre-trained [Clip](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) and [Dinov2](https://huggingface.co/facebook/dinov2-base/tree/main) models and place them in the `models/` directory:

```bash
mkdir -p models
```

If you want to use the Qwen model as the MLLM in the framework, you can download it yourself. The `model_manager.py` in the `utils` folder supports the Qwen model. If you wish to use other multimodal large models, you can simply modify the `model_manager.py` class accordingly.

### Directory Structure

```bash
├── data/                   
│   └── EvalRef-100/         # Content diverse dataset
├── models/                  # Pretrained models (CLIP, DINOv2)
├── benchmark_results/       # Generated images and intermediate results
├── src/                     # Core code
│   ├── benchmark.py         # Main program
│   ├── iut.py       		 # IUT class
│   ├── iter.py        		 # Iteration class
│   └── utils/       		 # Modelmanager
```

## 3. Running the Pipeline

### Basic Execution

Set models in benchmark.py before running:

```python
model_type = 'openai'       # Current support 'openai' and 'qwen'
sd_model_path = 'your_sd_model_path'
openai_api_key = 'your_api_key'

# openai model
model_manager.set_llm_model(model_type, openai_api_key)
# Qwen model
# model_manager.set_llm_model(model_type)


# Load stable diffusion models
model_manager.set_diffusion_model(sd_model_path)
# Load Clip & Dinov2 models
model_manager.set_vision_models()
```

Run the closed-loop evaluation with:

```bash
python benchmark.py
```

### Output Structure

After execution, results will be organized as:

```bash
benchmark_results/
├── example_run/
│   ├── example_final/            
│   │   ├── best_image.png
│   │   └── final_metrics.json    # Track prompt refinements
│   ├── iteration_1/            
│   │   ├── iter_1_best_image.png
│   │   └── iter_1_metadata.json   
│   ├── iteration_2/
│   │   ├── iter_1_best_image.png
│   │   └── iter_1_metadata.json
│   ├── ...
└── benchmark_final.json          # Benchmark results
```

## 4. Key Outputs

- **Visual Comparisons**: Side-by-side plots of reference vs. generated images
- **Quantitative Metrics**: JSON files containing:
  - CLIP similarity score
  - DINOv2 structural consistency
  - GPT4v content consistency and perceputual quality score (Creativity evaluation)
- **Prompt Evolution**: Full history of prompt refinements for reproducibility



# Dataset 

We present **EvalRef-100**, a meticulously curated benchmark dataset of 100 high-quality reference images designed to rigorously evaluate Text-to-Image (T2I) models. This dataset emphasizes **diversity**, **intentional challenges**, and **real-world complexity**, serving as a robust foundation for model assessment.

---

## **Dataset Composition**
| Category                  | Examples                                                     | Key Challenges                                               | Size |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| **1. Human Figures**      | - Multi-ethnic portraits<br>- Dynamic poses (dance/sports)<br>- Subtle expressions | Fine-grained attributes (clothing textures/skin tone gradients) | 25   |
| **2. Non-Human Entities** | - Rare objects (antique clocks)<br>- Multi-attribute compositions (glass with liquid refraction)<br>- Mixed materials (metal+fabric) | Physical plausibility/material details                       | 25   |
| **3. Scenes**             | - Perspective-rich interiors (museum domes)<br>- Multi-light nightscapes<br>- Dynamic elements (rainy streets with moving vehicles) | Spatial relationships/light interactions                     | 25   |
| **4. Complex Scenarios**  | - Counterintuitive scenes (floating mountains)<br>- Abstract art (surreal compositions)<br>- Culture-specific elements (traditional rituals) | Conceptual understanding/creative adaptation                 | 25   |

---

## **Core Advantages**
### 🎯 **Targeted Design**
- **Balanced Coverage**: Equal distribution across four critical domains  
- **Progressive Difficulty**: From single-object to multi-concept hybrids  

### 🏆 **High Quality**
- **Professional Curation**: Manually annotated metadata (object positions/materials/light directions)  

### 🧠 **Challenging Cases**
- **Adversarial Samples**: 10% images contain intentionally ambiguous elements, e.g.:  
  - *Paradoxical descriptions*: "Frozen flames" paired with volcanic ice caves  
  - *Long-Tail Objects*: Niche cultural artifacts (e.g., Inuit bone carving tools)  

---

## **Future Development**
The dataset will evolve through:  
1. **Continuous Expansion**: Ongoing integration of new edge cases and emerging challenge categories  
2. **Community-Driven Updates**: Collaborative refinement based on model failure analysis and user contributions  
3. **Dynamic Maintenance**: Regular updates to address evolving T2I model capabilities  

---

## **Access**  
Available under `data/Evalref-100/` with:  
- **CC-BY-NC 4.0** License  





## 🏫About us
Thank you for your interest in this project. The project is supervised by the DCD Lab at Zhejiang University’s College of Computer Science and Technology.

## Contact us
If you have any questions, feel free to contact me via email 12321215@zju.edu.cn.

## Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```
@misc{meng2024imageregenerationevaluatingtexttoimage,
      title={Image Regeneration: Evaluating Text-to-Image Model via Generating Identical Image with Multimodal Large Language Models}, 
      author={Chutian Meng and Fan Ma and Jiaxu Miao and Chi Zhang and Yi Yang and Yueting Zhuang},
      year={2024},
      eprint={2411.09449},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.09449}, 
}
```
