import torch
import openai
from typing import Optional, Union, List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from diffusers import StableDiffusionPipeline
from modelscope import AutoModelForCausalLM, AutoTokenizer
import base64
from pathlib import Path
from io import BytesIO

class ModelManager:
    """
    Global model manager supporting multiple model types and configurations
    Handles Qwen, OpenAI, CLIP, DINOv2, and Stable Diffusion models
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize manager state"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.llm_model_type = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.openai_key = None
        self.clip_model = None
        self.clip_processor = None
        self.dinov2_model = None
        self.dinov2_processor = None
        self.sd_model = None  # Current SD model
        self.sd_model_name = None

    def set_llm_model(self, model_type: str, **kwargs):
        """
        Set large language model configuration
        Args:
            model_type: "qwen" or "openai"
            kwargs: openai_key (required for OpenAI)
        """
        if model_type not in ["qwen", "openai"]:
            raise ValueError("Invalid model type. Choose 'qwen' or 'openai'")
        
        self.llm_model_type = model_type
        
        if model_type == "openai":
            if "openai_key" not in kwargs:
                raise ValueError("openai_key is required for OpenAI")
            self.openai_key = kwargs["openai_key"]
            openai.api_key = self.openai_key
        elif model_type == "qwen":
            self._load_qwen()

    def set_diffusion_model(self, model_name: str):
        """
        Set/load Stable Diffusion model
        Clears previous model from memory if exists
        """
        if self.sd_model:
            del self.sd_model
            torch.cuda.empty_cache()
        
        self.sd_model = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device)
        self.sd_model_name = model_name

    def set_vision_models(self):
        """Load shared vision models"""
        if self.clip_model is None:
            print("Loading CLIP model...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        if self.dinov2_model is None:
            print("Loading DINOv2 model...")
            self.dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)

    def _load_qwen(self):
        """Load Qwen-VL model components"""
        if self.qwen_model is None:
            print("Loading Qwen-VL model...")
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                "qwen/Qwen-VL-Chat",
                trust_remote_code=True
            )
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                "qwen/Qwen-VL-Chat",
                device_map=self.device,
                trust_remote_code=True
            )
            

    def get_diffusion_model(self) -> StableDiffusionPipeline:
        """Get current Stable Diffusion model instance"""
        if not self.sd_model:
            raise ValueError("Diffusion model not loaded")
        return self.sd_model
    
    def get_clip_model(self):
        if self.clip_model == None:
            raise ValueError("Clip model not initialized")
        return self.clip_model, self.clip_processor
    
    def get_dinov2_model(self):
        if self.dinov2_model == None:
            raise ValueError("Dinov2 model not initialized")
        return self.dinov2_model, self.dinov2_processor

    def call_llm(self, 
                prompt: str, 
                image_paths: Optional[Union[str, List[str]]] = None) -> str:
        """
        Call LLM with text and optional images
        Args:
            prompt: Text prompt
            image_paths: Single path or list of image paths
        Returns:
            Generated text response
        """
        if self.llm_model_type == "qwen":
            return self._call_qwen(prompt, image_paths)
        elif self.llm_model_type == "openai":
            return self._call_openai(prompt, image_paths)
        else:
            raise ValueError("LLM not initialized")

    def _call_qwen(self, prompt: str, image_path: Optional[Union[str, List[str]]]) -> str:
        """Call local Qwen-VL model with optional image inputs.
        
        Args:
            prompt (str): The text prompt to send to the model.
            image_paths (list, optional): A list of image paths. Defaults to None.
        
        Returns:
            str: The response from the model.
        """
        query_list = []
        
        # Add images to the query list if provided
        if image_path:
            if isinstance(image_path, str):
                image_path = [image_path]

            for path in image_path:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Image not found: {path}")
                query_list.append({'image': path})
        
        # Add the text prompt to the query list
        query_list.append({'text': prompt})
        print(query_list)
        # Convert the query list to the format expected by the tokenizer
        query = self.qwen_tokenizer.from_list_format(query_list)
        
        # Call the model with the constructed query
        response, _ = self.qwen_model.chat(self.qwen_tokenizer, query=query, history=None)
        print(response)
        return response

    def _call_openai(self, prompt: str, image_path: Optional[Union[str, List[str]]]) -> str:
        """Call OpenAI model"""
        messages = [{"type": "text", "text": prompt}]

        if image_path:
            if isinstance(image_path, str):
                image_path = [image_path]
                
            images = []
            for path in image_path:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Image not found: {path}")
                    
                with Image.open(path) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    images.append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                    })
            
            messages.extend(images)
            
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview" if image_path else "gpt-4",
            messages=[{"role": "user", "content": messages}],
            max_tokens=1000
        )
        return response.choices[0].message.content

    
    

# Global singleton instance
model_manager = ModelManager()