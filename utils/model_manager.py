# model_manager.py
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from diffusers import StableDiffusionPipeline
from typing import Dict, List, Optional

class ModelManager:
    """
    Global model management class, supporting：
    - Qwen-VL
    - CLIP
    - DINOv2
    - Stable Diffusion models
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_models()
        return cls._instance
    
    def _initialize_models(self):
        """Initialize models"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = None  # （qwen/openai）
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.clip_model = None
        self.clip_processor = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.sd_models = {}  # Stable Diffusion model cache
    
    def set_model_type(self, model_type: str):
        """
        Args:
            model_type: "qwen" or "openai"
        """
        if model_type not in ["qwen", "openai"]:
            raise ValueError("model_type must be 'qwen' or 'openai'")
        self.model_type = model_type
        
        if model_type == "qwen":
            self._load_qwen()
        
        self._load_clip()
        self._load_dinov2()
        
    def _load_qwen(self):
        """Load Qwen-VL model"""
        print("Loading Qwen-VL model...")
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "qwen/Qwen-VL-Chat",
            device_map="auto",
            trust_remote_code=True
        )
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            "qwen/Qwen-VL-Chat",
            trust_remote_code=True
        )
    
    def _load_clip(self):
        """Load CLIP model"""
        print("Loading CLIP model...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
    
    def _load_dinov2(self):
        """Load DINOv2 model"""
        print("Loading DINOv2 model...")
        self.dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
    
    def get_qwen(self):
        """Get Qwen model"""
        return self.qwen_model, self.qwen_tokenizer
    
    def get_clip(self):
        """Get CLIP model"""
        return self.clip_model, self.clip_processor
    
    def get_dinov2(self):
        """Get DINOv2 model"""
        return self.dinov2_model, self.dinov2_processor
    
    def get_sd_model(self, model_name: str) -> StableDiffusionPipeline:
        """
        Get Stable Diffusion model
        """
        if model_name not in self.sd_models:
            print(f"Loading Stable Diffusion model: {model_name}...")
            self.sd_models[model_name] = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
            ).to(self.device)
        return self.sd_models[model_name]
    
    def clear_sd_cache(self):
        """Clear Stable Diffusion cache"""
        self.sd_models.clear()
        torch.cuda.empty_cache()


model_manager = ModelManager()