import json
from pathlib import Path
from typing import Dict
from pathlib import Path
import re

from utils.model_manager import model_manager


class IUTGenerator:
    """
    Image Understanding Tree Generator with multi-model support
    """
    
    def __init__(self):
        """
        Initialize IUT generator
        """
        
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
        "global_description": """
            Analyze the provided image and generate a single, comprehensive sentence describing its content and style. The description must:  
                1. Begin with the primary subject or scene  
                2. Include key visual elements (objects, colors, lighting)  
                3. Mention the artistic style or visual characteristics (e.g., realistic, abstract, vibrant, muted)  
                4. End with the overall mood or atmosphere (e.g., serene, chaotic, nostalgic)  
                5. Be concise and grammatically correct  
            Output only the description sentence, without any additional text, explanations, or formatting.  
        """,
        
        "global_features": """
            Analyze the provided image and extract its overall visual characteristics. Output the results in a strict JSON format with the following fields and descriptions:
                {
                    "style": "Artistic style (e.g., photorealistic, cartoonish)",
                    "lighting": "Lighting characteristics (e.g., soft natural light, dramatic shadows)",
                    "color_palette": "Dominant colors and scheme (e.g., warm pastel, cool monochrome)",
                    "composition": "Composition technique (e.g., rule of thirds, symmetry, leading lines)"
                }
            Output only the JSON object without any additional text, explanations, or formatting. Ensure the JSON is valid and properly structured.
        """,
        
        "object_detection": """
            Analyze the provided image and identify the main objects (e.g., people, animals, or objects). List at most 5 objects. Output the results in the following strict JSON format:
                {
                    "objects": [
                        {
                            "name": "Descriptive name (e.g., smiling child, wooden chair)",
                            "type": "person/animal/object"
                        }
                    ]
                }
            Output only the JSON object without any additional text, explanations, or formatting. Ensure the JSON is valid and properly structured.
        """,
        
        "object_details": """
            Analyze the object named {obj_name} in the provided image and extract its detailed characteristics. Output the results in the following strict JSON format:
                {{
                    "color": "Main color(s) (e.g., red, blue, metallic silver)",
                    "material": "Material composition (e.g., wood, plastic, fabric)",
                    "position": "Position in frame (e.g., center, top right, bottom left)",
                    "texture": "Surface texture (e.g., smooth, rough, glossy)",
                    {type_specific}
                }}
            Output only the JSON object without any additional text, explanations, or formatting. Ensure the JSON is valid and properly structured.
        """,
        
        "relationships": """
            Analyze the relationships between the main objects in the provided image and output the results in the following strict JSON format:
                {
                    "relationships": [
                        "subject verb object (e.g., man holding cup)",
                        "spatial relationships (e.g., cat on sofa)"
                    ]
                }
            Output only the JSON object without any additional text, explanations, or formatting. Ensure the JSON is valid and properly structured.
        """
    }

    TYPE_SPECIFIC_ATTRIBUTES = {
        "person": """
            "posture": "Body posture (e.g., standing, sitting)",
            "expression": "Facial expression (e.g., smiling, neutral)",
            "clothing": "Clothing description (e.g., red jacket, blue jeans)
        """,
        "animal": """
            "posture": "Body posture (e.g., sitting, running)",
            "species": "Animal species (e.g., golden retriever, tabby cat)",
            "expression": "Facial expression (e.g., alert, relaxed)"
        """,
        "object": """
            "function": "Object purpose (e.g., seating, storage)",
            "state": "Condition state (e.g., new, worn, damaged)"
        """
    }

    def set_reference_image(self, image_path: str):
        """Load and validate reference image"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        self.ref_image_path = image_path

    def _parse_json_response(self, response: str) -> Dict:
        """Strict JSON parsing with validation"""
        try:
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response, re.DOTALL)
            json_str = match.group(0)
            json_data = json.loads(json_str)
            if not isinstance(json_data, dict):
                raise ValueError("Response is not a JSON object")
            return json_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}")

    def generate_iut(self) -> Dict:
        """Generate complete Image Understanding Tree"""
        if self.ref_image_path is None:
            raise ValueError("Reference image not set")
        
        try:
            # Step 1: Global description
            global_desc = model_manager.call_llm(self.PROMPT_TEMPLATES["global_description"], self.ref_image_path)
            self.iut["global_description"] = global_desc.strip('"')
            
            # Step 2: Global features
            features = self._parse_json_response(
                model_manager.call_llm(self.PROMPT_TEMPLATES["global_features"], self.ref_image_path)
            )
            self.iut["global_features"].update(features)
            
            # Step 3: Object detection
            objects_data = self._parse_json_response(
                model_manager.call_llm(self.PROMPT_TEMPLATES["object_detection"], self.ref_image_path)
            )
            
            # Step 4: Object details
            for obj in objects_data["objects"]:
                prompt = self.PROMPT_TEMPLATES["object_details"].format(
                    obj_name=obj["name"],
                    type_specific=self.TYPE_SPECIFIC_ATTRIBUTES[obj["type"]]
                )
                attributes = self._parse_json_response(model_manager.call_llm(prompt, self.ref_image_path))
                
                self.iut["objects"].append({
                    "name": obj["name"],
                    "type": obj["type"],
                    "attributes": attributes
                })
            
            # Step 5: Relationships
            if len(self.iut["objects"]) > 1:
                relationships = self._parse_json_response(
                    model_manager.call_llm(self.PROMPT_TEMPLATES["relationships"], self.ref_image_path)
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
        Transform the following original prompt into a well-structured Stable Diffusion prompt. Ensure the output:
        1. Maintains all key information from the original prompt.
        2. Follows the standard Stable Diffusion format: 
        - Use commas to separate descriptive elements.
        - Prioritize the most important details first.
        - Include style, subject, and context descriptors.
        - Use concise and vivid language.
        3. Output only the transformed Stable Diffusion prompt, without any additional text or explanations.

        Original prompt: {original_prompt}
    """

    def __init__(self, 
                 iut_data: Dict):
        """
        Initialize prompt generator
        
        Args:
            iut_data: Image Understanding Tree structure
            model_type: "openai" or "qwen"
            openai_api_key: Required for OpenAI
            qwen_model: Preloaded Qwen model instance
        """
        self.iut = iut_data

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
        optimization_prompt = self.OPTIMIZATION_PROMPT.format(original_prompt=base_prompt)
        return model_manager.call_llm(optimization_prompt)

    def run(self) -> str:
        """Full generation pipeline"""
        base_prompt = self.generate_base_prompt()
        return self.optimize_prompt(base_prompt)

    

   



if __name__ == "__main__":
    
    model_manager.set_llm_model(model_type='openai', openai_key='your_api_key')
    # model_manager.set_llm_model(model_type="qwen", model_path='your_qwen_path')
    iut_generator = IUTGenerator()
    iut_generator.set_reference_image("your_img_path")
    iut_data = iut_generator.generate_iut()
    iut_generator.save_iut("output_iut.json")

    prompt_generator = PromptGenerator(iut_data=iut_data)
    optimized_prompt = prompt_generator.run()
    print("Optimized Prompt:")
    print(optimized_prompt)

    