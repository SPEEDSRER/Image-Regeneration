import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from pathlib import Path

from utils.model_manager import model_manager


class IUTGenerator:
    """
    Image Understanding Tree Generator with multi-model support
    """
    
    def __init__(self):
        """
        Initialize IUT generator
        """
        
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
        self.ref_image_path = image_path

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
            global_desc = model_manager.call_llm(self.PROMPT_TEMPLATES["global_description"], self.ref_image_path)
            self.iut["global_description"] = global_desc.strip('"')
            
            # Step 2: Global features
            features = self._parse_json_response(
                self.model_manager.call_llm(self.PROMPT_TEMPLATES["global_features"], self.ref_image_path)
            )
            self.iut["global_features"].update(features)
            
            # Step 3: Object detection
            objects_data = self._parse_json_response(
                self.model_manager.call_llm(self.PROMPT_TEMPLATES["object_detection"], self.ref_image_path)
            )
            
            # Step 4: Object details
            for obj in objects_data["objects"]:
                prompt = self.PROMPT_TEMPLATES["object_details"].format(
                    obj_name=obj["name"],
                    type_specific=self.TYPE_SPECIFIC_ATTRIBUTES[obj["type"]]
                )
                attributes = self._parse_json_response(self.model_manager.call_llm(prompt, self.ref_image_path))
                
                self.iut["objects"].append({
                    "name": obj["name"],
                    "type": obj["type"],
                    "attributes": attributes
                })
            
            # Step 5: Relationships
            if len(self.iut["objects"]) > 1:
                relationships = self._parse_json_response(
                    self.model_manager.call_llm(self.PROMPT_TEMPLATES["relationships"], self.ref_image_path)
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
        print(base_prompt)
        optimization_prompt = self.OPTIMIZATION_PROMPT.format(prompt=base_prompt)
        return model_manager.call_llm(optimization_prompt)

    def run(self) -> str:
        """Full generation pipeline"""
        base_prompt = self.generate_base_prompt()
        return self.optimize_prompt(base_prompt)

    

   



if __name__ == "__main__":
    
    iut_generator = IUTGenerator()
    iut_generator.set_reference_image("test.jpg")
    iut_data = iut_generator.generate_iut()
    iut_generator.save_iut("output_openai.json")

    prompt_generator = PromptGenerator(iut_data=iut_data)
    optimized_prompt = prompt_generator.run()
    print("Qwen Optimized Prompt:")
    print(optimized_prompt)

    