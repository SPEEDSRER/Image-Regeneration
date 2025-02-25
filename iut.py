import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import openai
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from utils.model_manager import model_manager

# model_id = 'qwen/Qwen-VL-Chat'
# revision = 'v1.1.0'
# print('Using model:',model_id)
# model_path = snapshot_download(model_id, revision=revision)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# global_qwen_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# global_qwen_model = AutoModelForCausalLM.from_pretrained(
#     model_path, 
#     device_map=device, 
#     trust_remote_code=True
# ).eval()



class IUTGenerator:
    """
    Image Understanding Tree Generator with multi-model support
    """
    
    # Qwen model should be loaded externally like:
    # from modelscope import AutoModelForCausalLM
    # qwen_model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-VL-Chat", device_map="auto")
    
    def __init__(self, 
                 model_type: str = "openai",
                 openai_api_key: str = None,):
        """
        Initialize IUT generator
        
        Args:
            model_type: "openai" or "qwen"
            openai_api_key: Required for OpenAI
            qwen_model: Preloaded Qwen model instance
        """
        self.model_type = model_type
        if model_type == 'openai':
            if openai_api_key == None:
                raise ValueError("Missing openai API key")
            else:
                self.openai_api_key = openai_api_key
        elif model_type == 'qwen':
            self.qwen_model = model_manager.get_qwen_model()
            self.qwen_tokenizer = model_manager.get_qwen_tokenizer()
        else:
            raise ValueError("Unsupported model type")
        
        
        self.ref_image = None
        self.ref_image_path = None
        self.iut = {
            "global_description": "",
            "global_features": {
                "style": "",
                "lighting": "",
                "color_palette": "",
                "composition": ""
            },
            "objects": [],
            "relationships": []
        }

    # Core prompt templates
    PROMPT_TEMPLATES = {
        "global_description": """Generate a comprehensive one-sentence description of this image focusing on:
1. Main subject and key objects
2. Overall atmosphere and style
3. Spatial relationships""",
        
        "global_features": """Analyze and output JSON with these keys:
{
    "style": "Artistic style (e.g., photorealistic, cartoonish)",
    "lighting": "Lighting characteristics (e.g., soft natural light)",
    "color_palette": "Dominant colors and scheme (e.g., warm pastel)",
    "composition": "Composition technique (e.g., rule of thirds)"
}""",
        
        "object_detection": """List main objects in JSON format:
{
    "objects": [
        {
            "name": "Descriptive name",
            "type": "person/animal/object",
            "primary_attributes": ["color", "position"]
        }
    ]
}""",
        
        "object_details": """Analyze {obj_name} and output JSON with:
{{
    "color": "Main color(s)",
    "material": "Material composition",
    "position": "Position in frame",
    "texture": "Surface texture",
    {type_specific}
}}""",
        
        "relationships": """Identify relationships between objects in JSON:
{
    "relationships": [
        "subject verb object (e.g., man holding cup)",
        "spatial relationships (e.g., cat on sofa)"
    ]
}"""
    }

    TYPE_SPECIFIC_ATTRIBUTES = {
        "person": '"posture": "Body posture","expression": "Facial expression", "clothing": "Clothing description"',
        "animal": '"posture": "Body posture", "species": "Animal species", "expression": "Facial expression"',
        "object": '"function": "Object purpose", "state": "Condition state"'
    }

    def set_reference_image(self, image_path: str):
        """Load and validate reference image"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            self.ref_image = Image.open(path).convert("RGB")
            self.ref_image_path = image_path
            self.ref_image.verify()
        except Exception as e:
            raise ValueError(f"Invalid image: {str(e)}")

    def _encode_image(self) -> str:
        """Encode image to base64"""
        buffered = base64.b64encode(self.ref_image.tobytes())
        return buffered.decode("utf-8")

    def _call_model(self, prompt: str) -> str:
        """Unified model calling interface"""
        if self.model_type == "openai":
            return self._call_openai(prompt)
        elif self.model_type == "qwen":
            return self._call_qwen(prompt)
        else:
            raise ValueError("Unsupported model type")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT-4 Vision"""
        openai.api_key = self.openai_api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", 
                     "image_url": f"data:image/jpeg;base64,{self._encode_image()}"}
                ]
            }],
            max_tokens=1024
        )
        return response.choices[0].message.content

    def _call_qwen(self, prompt: str) -> str:
        """Call local Qwen-VL model"""
        query = self.qwen_tokenizer.from_list_format([
            {'image': self.ref_image_path},
            {'text': prompt}
        ])
        response, _ = self.qwen_model.chat(self.qwen_tokenizer, query=query, history=None)
        print(response)
        return response

    def _parse_json_response(self, response: str) -> Dict:
        """Strict JSON parsing with validation"""
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")

    def generate_iut(self) -> Dict:
        """Generate complete Image Understanding Tree"""
        if self.ref_image is None:
            raise ValueError("Reference image not set")
        
        try:
            # Step 1: Global description
            global_desc = self._call_model(self.PROMPT_TEMPLATES["global_description"])
            self.iut["global_description"] = global_desc.strip('"')
            
            # Step 2: Global features
            features = self._parse_json_response(
                self._call_model(self.PROMPT_TEMPLATES["global_features"])
            )
            self.iut["global_features"].update(features)
            
            # Step 3: Object detection
            objects_data = self._parse_json_response(
                self._call_model(self.PROMPT_TEMPLATES["object_detection"])
            )
            
            # Step 4: Object details
            for obj in objects_data["objects"]:
                prompt = self.PROMPT_TEMPLATES["object_details"].format(
                    obj_name=obj["name"],
                    type_specific=self.TYPE_SPECIFIC_ATTRIBUTES[obj["type"]]
                )
                attributes = self._parse_json_response(self._call_model(prompt))
                
                self.iut["objects"].append({
                    "name": obj["name"],
                    "type": obj["type"],
                    "attributes": attributes
                })
            
            # Step 5: Relationships
            if len(self.iut["objects"]) > 1:
                relationships = self._parse_json_response(
                    self._call_model(self.PROMPT_TEMPLATES["relationships"])
                )
                self.iut["relationships"] = relationships["relationships"]
            else:
                self.iut["relationships"] = []
            
            return self.iut
        
        except Exception as e:
            raise RuntimeError(f"IUT generation failed: {str(e)}")

    def save_iut(self, output_path: str):
        """Save IUT to JSON file"""
        with open(output_path, "w") as f:
            json.dump(self.iut, f, indent=2)



class PromptGenerator:
    """
    Stable Diffusion Prompt Generator with multi-model support
    """
    
    PROMPT_TEMPLATE = """
    {style} style, {color_palette}, {composition},
    {main_description},
    {objects_section}
    {relationships_section}
    """
    
    OBJECT_TEMPLATE = {
        "person": "{attributes} {name} {posture} with {expression} expression wearing {clothing}",
        "animal": "{attributes} {name} ({species}) {posture} with {expression}",
        "object": "{material} {color} {name} in {state} state used for {function}"
    }

    OPTIMIZATION_PROMPT = """
    Optimize this prompt for Stable Diffusion following these rules:
    1. Use comma-separated phrases
    2. Put important elements first
    3. Add relevant details like lighting and textures
    4. Use specific artistic terms
    5. Keep under 300 characters
    
    Original prompt: {prompt}
    Optimized prompt:"""

    def __init__(self, 
                 iut_data: Dict,
                 model_type: str = "openai",
                 openai_api_key: str = None):
        """
        Initialize prompt generator
        
        Args:
            iut_data: Image Understanding Tree structure
            model_type: "openai" or "qwen"
            openai_api_key: Required for OpenAI
            qwen_model: Preloaded Qwen model instance
        """
        self.iut = iut_data
        self.model_type = model_type
        if model_type == 'openai':
            if openai_api_key == None:
                raise ValueError("Missing openai API key")
            else:
                self.openai_api_key = openai_api_key
        elif model_type == 'qwen':
            self.qwen_model = model_manager.get_qwen_model()
            self.qwen_tokenizer = model_manager.get_qwen_tokenizer()
        else:
            raise ValueError("Unsupported model type")


    def _call_model(self, prompt: str) -> str:
        """Unified model calling interface"""
        if self.model_type == "openai":
            return self._call_openai(prompt)
        elif self.model_type == "qwen":
            return self._call_qwen(prompt)
        else:
            raise ValueError("Unsupported model type")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT-4"""
        openai.api_key = self.openai_api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a prompt optimization expert"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()

    def _call_qwen(self, prompt: str) -> str:
        """Call local Qwen model"""
        query = self.qwen_tokenizer.from_list_format([
            {'text': prompt}
        ])
        response, _ = self.qwen_model.chat(self.qwen_tokenizer, query=query, history=None)
        print(response)
        return response

    def _process_global_features(self) -> Dict[str, str]:
        """Extract and format global features"""
        gf = self.iut["global_features"]
        return {
            "style": gf["style"],
            "color_palette": gf["color_palette"],
            "composition": gf["composition"]
        }

    def _process_objects(self) -> str:
        """Generate detailed objects description"""
        object_descriptions = []
        
        for obj in self.iut["objects"]:
            obj_type = obj["type"]
            template = self.OBJECT_TEMPLATE[obj_type]
            
            # Format attributes
            attributes = []
            for key, value in obj["attributes"].items():
                if key in ["posture", "expression", "clothing", "species", "state", "function"]:
                    continue
                attributes.append(f"{value} {key}")
                
            filled = template.format(
                name=obj["name"],
                attributes=", ".join(attributes),
                **obj["attributes"]
            )
            object_descriptions.append(filled)
        
        return ", ".join(object_descriptions)

    def _process_relationships(self) -> str:
        """Format relationships section"""
        return ", ".join(self.iut["relationships"])

    def generate_base_prompt(self) -> str:
        """Generate initial prompt from IUT"""
        components = {
            "style": self.iut["global_features"]["style"],
            "color_palette": self.iut["global_features"]["color_palette"],
            "composition": self.iut["global_features"]["composition"],
            "main_description": self.iut["global_description"],
            "objects_section": self._process_objects(),
            "relationships_section": self._process_relationships()
        }
        
        return self.PROMPT_TEMPLATE.format(**components).strip()

    def optimize_prompt(self, base_prompt: str) -> str:
        """Optimize prompt using LLM"""
        print(base_prompt)
        optimization_prompt = self.OPTIMIZATION_PROMPT.format(prompt=base_prompt)
        return self._call_model(optimization_prompt)

    def run(self) -> str:
        """Full generation pipeline"""
        base_prompt = self.generate_base_prompt()
        return self.optimize_prompt(base_prompt)

    

   



if __name__ == "__main__":
    # For OpenAI
    openai_generator = IUTGenerator(
        model_type="openai",
        openai_api_key="your_openai_key"
    )
    openai_generator.set_reference_image("test.jpg")
    iut_data = openai_generator.generate_iut()
    openai_generator.save_iut("output_openai.json")

    # # For Qwen (pseudo implementation)
    # qwen_generator = IUTGenerator(
    #     model_type="qwen",
    #     openai_api_key = None
    # )
    # qwen_generator.set_reference_image("")
    # iut_data = qwen_generator.generate_iut()
    # qwen_generator.save_iut("output_qwen.json")

    # qwen_generator = PromptGenerator(
    #     iut_data=iut_data,
    #     model_type="qwen",
    #     openai_api_key = None
    # )
    # optimized_prompt = qwen_generator.run()
    # print("Qwen Optimized Prompt:")
    # print(optimized_prompt)