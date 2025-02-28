
# [ AAAI2025 ] Image Regeneration: Evaluating Text-to-Image Model via Generating Identical Image with Multimodal Large Language Models
### [[Paper]](https://arxiv.org/abs/2411.09449)

## 🔥 News 

- 2025-02-28: We have release the code for Image Regeneration, supporting qwen-vl & GPT

## To Do List
- [x] Code
- [x] Dataset
- [ ] Project Page
- [ ] More testing results via different MLLMs

## Installation

### Conda environment setup
```
conda create -n ImageRepainter python=3.10.14
conda activate ImageRepainter
pip install -r requirement.txt
```
### Required models
Download the Clip & Dinov2 model files and put them in the vision_models folder.
```
├── Image_Regen
│   ├── vision_models
│   ├──   ├── Clip
│   ├──   ├── Dinov2
├── ├── Qwen(Optional)
│   ├──   ├── qwen-vl
│   ├── ...
```
## Set models
We have designed a Model_Manager class for the convenience of utilizing multiple models. Before running the benchmark, set the models required for the experiment.
```python
model_type = ''     # Current support 'openai' and 'qwen'
sd_model_path = 'your_sd_model_path'
openai_api_key = 'your_api_key'

# Qwen model
model_manager.set_llm_model(model_type)
# GPT model
# model_manager.set_llm_model(model_type, openai_api_key)

# Load stable diffusion models
model_manager.set_diffusion_model(sd_model_path)
# Load clip & dinov2 models
model_manager.set_vision_models()
```


## Run benchmark


After setting the required models,  you can quickly run the benchmark on diffusion model by using the following command,
```
CUDA_VISIBLE_DEVICES=0 python benchmark.py
```



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
