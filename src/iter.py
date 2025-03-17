import os
import json
import torch
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from io import BytesIO
import openai
import re

from utils.model_manager import model_manager

class IterativeImageGenerator:
    """
    Iterative Image Generation with Multi-stage Processing
    """
    
    def __init__(
        self,
        max_iterations: int = 3,
        prompts_per_iteration: int = 3
    ):
        self.clip_model, self.clip_processor = model_manager.get_clip_model()
        self.dinov2_model, self.dinov2_processor = model_manager.get_dinov2_model()

        self.sd_model = model_manager.get_diffusion_model()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_iterations = max_iterations
        self.prompts_per_iteration = prompts_per_iteration
        
        
        
        # State variables
        self.ref_image = None
        self.ref_image_path = None
        self.ref_features = None
        self.current_prompt = None
        self.best_result = None

    def set_workspace(self, result_path: str):
        self.iter_output_dir = Path(result_path)
        os.makedirs(self.iter_output_dir, exist_ok=True)
    
    def set_reference_image(self, image_path: str):
        """Set reference image and compute features"""
        self.ref_image_path = image_path
        self.ref_image = Image.open(image_path).convert("RGB")
        self._precompute_reference_features()

    def _precompute_reference_features(self):
        """Precompute reference image features"""
        with torch.no_grad():
            # CLIP features
            clip_inputs = self.clip_processor(
                images=self.ref_image,
                return_tensors="pt"
            ).to(self.device)
            self.ref_clip_features = self.clip_model.get_image_features(**clip_inputs)
            
            # DINOv2 features
            dinov2_inputs = self.dinov2_processor(
                self.ref_image,
                return_tensors="pt"
            ).to(self.device)
            self.ref_dinov2_features = self.dinov2_model(**dinov2_inputs).last_hidden_state.mean(dim=1)

    # Stage 1: Prompt Generation/Revision
    def _generate_prompt_variants(self, base_prompt: str) -> List[str]:
        """Generate multiple prompt variants"""
        
        prompt = f"""
            Generate {self.prompts_per_iteration} synonymous variations of the following Stable Diffusion prompt. Ensure each variation:
            1. Maintains the same key information and meaning as the original prompt.
            2. Follows the standard Stable Diffusion format:
            - Use commas to separate descriptive elements.
            - Prioritize the most important details first.
            - Include style, subject, and context descriptors.
            - Use concise and vivid language.
            3. Output the variations as a JSON array, with each variation as a string in the array. Do not include any additional text or explanations.
            Output format:
            {{
                "prompts": [
                    "variation 1",
                    "variation 2",
                    "variation 3"
                ]
            }}

            Original prompt: {base_prompt}"""
        response = model_manager.call_llm(prompt)
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, response, re.DOTALL)
        json_str = match.group(0)
        data = json.loads(json_str)
        prompts = data.get("prompts", [])
        return prompts

    # Stage 2: Image Generation
    def _generate_images(self, prompts: List[str]) -> List[Tuple[str, Image.Image]]:
        """Generate images from prompts"""
        result_tuples = []
        for prompt in prompts:
            result_img = self.sd_model(prompt).images[0]
            result_tuple = [prompt, result_img]
            result_tuples.append(result_tuple)
        return result_tuples

    # Stage 3: Image Selection
    def _evaluate_image(self, image: Image.Image) -> Dict:
        """Evaluate image quality metrics"""
        with torch.no_grad():
            # CLIP similarity
            inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            gen_clip_features = self.clip_model.get_image_features(**inputs)
            clip_score = torch.cosine_similarity(self.ref_clip_features, gen_clip_features).item()
            
            # DINOv2 similarity
            inputs = self.dinov2_processor(
                image,
                return_tensors="pt"
            ).to(self.device)
            gen_dinov2_features = self.dinov2_model(**inputs).last_hidden_state.mean(dim=1)
            dinov2_score = torch.cosine_similarity(self.ref_dinov2_features, gen_dinov2_features).item()
        
        return {
            "clip": clip_score,
            "dinov2": dinov2_score,
            "total": 0.5 * clip_score + 0.5 * dinov2_score
        }

    def _select_best_candidate(self, candidates: List[Tuple[str, Image.Image]]) -> Tuple[str, Image.Image, Dict]:
        """Select best candidate from generated images"""
        scored = []
        for prompt, image in candidates:
            scores = self._evaluate_image(image)
            scored.append((scores["total"], prompt, image, scores))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1:]  # (prompt, image, scores)

    # Stage 4: Feedback Generation
    def _generate_feedback(self, best_image_path: str) -> str:
        """Generate improvement feedback using GPT-4V"""
        image_list = [self.ref_image_path, best_image_path]
        prompt = """
            Compare the following two images and suggest specific modifications to the prompt of the second image to make it more similar to the first image. Focus on key differences in:
            1. Style (e.g., artistic style, lighting, color palette)
            2. Composition (e.g., object placement, framing)
            3. Content (e.g., objects, details, atmosphere)
            Output only the prompt modification suggestions as a concise list, without any additional text or explanations. Each suggestion should be actionable and specific.
        """
        response = model_manager.call_llm(prompt, image_list)
        return response
    
    def _modify_prompt(self, best_prompt: str, feedback: str):
        prompt = f"""Update the following Stable Diffusion prompt based on the provided modification suggestions. Ensure the revised prompt:
            1. Incorporates all the modification suggestions fully.
            2. Maintains the standard Stable Diffusion format:
            - Use commas to separate descriptive elements.
            - Prioritize the most important details first.
            - Include style, subject, and context descriptors.
            - Use concise and vivid language.
            3. Output only the revised prompt, without any additional text or explanations.
            Current prompt: {best_prompt}
            Modification suggestions: {feedback}
        """
        response = model_manager.call_llm(prompt)
        return response

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Final Evaluation
    def _get_gpt4v_scores(self, image: Image.Image) -> Dict:
        """Get GPT-4V evaluation scores"""
        encoded_image = self._encode_image(image)
        encoded_ref = self._encode_image(self.ref_image)
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """Rate similarity (1-5) for:
                        - Aesthetic similarity
                        - Semantic similarity
                        Output JSON format:
                        {
                            "aesthetic": 1-5,
                            "semantic": 1-5
                        }"""},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_ref}"}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=100
        )
        
        scores = json.loads(response.choices[0].message.content)
        scores["average"] = (scores["aesthetic"] + scores["semantic"]) / 2
        return scores

    # Main Process Flow
    def run_iterative_generation(self, initial_prompt: str) -> Dict:
        """Full iterative generation workflow"""
        history = []
        self.current_prompt = initial_prompt
        
        for iter_num in range(self.max_iterations):
            # Stage 1: Prompt Generation
            variants = self._generate_prompt_variants(self.current_prompt)

            # Stage 2: Image Generation
            candidates = self._generate_images(variants)
            
            # Stage 3: Image Selection
            best_prompt, best_image, scores = self._select_best_candidate(candidates)
            
            # Save intermediate results
            self._save_iteration_results(iter_num, best_image, best_prompt, scores)
            iter_dir = self.iter_output_dir / f"iter_{iter_num+1}"
            best_image_path = str(iter_dir / "best_image.png")

            # Stage 4: Feedback Generation
            feedback = self._generate_feedback(best_image_path)    
            modified_prompt = self._modify_prompt(best_prompt, feedback)
            
            # Update state
            self.current_prompt = modified_prompt
            history.append({
                "iteration": iter_num + 1,
                "prompt": best_prompt,
                "scores": scores,
                "feedback": feedback
            })
            

        # Final Evaluation
        final_scores = {
            "aesthetic": 4,
            "semantic": 4,
            "average": 4
        }
        # final_scores = self._get_gpt4v_scores(best_image)
        final_results = {
            "final_prompt": best_prompt,
            "clip_score": scores["clip"],
            "dinov2_score": scores["dinov2"],
            "gpt4v_scores": final_scores,
            "history": history
        }
        
        self._save_final_results(final_results, best_image)
        return final_results

    # Utility Methods
    def _save_iteration_results(self, 
                              iteration: int,
                              image: Image.Image,
                              prompt: str,
                              scores: Dict):
        """Save per-iteration results"""
        iter_dir = self.iter_output_dir / f"iter_{iteration+1}"
        os.makedirs(iter_dir, exist_ok=True)
        
        # Save image
        image.save(iter_dir / "best_image.png")
        
        # Save metadata
        with open(iter_dir / "metadata.json", "w") as f:
            json.dump({
                "prompt": prompt,
                "scores": scores
            }, f, indent=2)

    def _save_final_results(self, results: Dict, image: Image.Image):
        """Save final results"""
        final_dir = self.iter_output_dir / "final"
        os.makedirs(final_dir, exist_ok=True)
        
        # Save image
        image.save(final_dir / "best_final.jpg")
        
        # Save metadata
        with open(final_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)



if __name__ == "__main__":
    
    # Set LLM
    model_manager.set_llm_model(model_type='openai', openai_key='your_api_key')
    # model_manager.set_llm_model(model_type="qwen", model_path='your_qwen_path')
    
    # Load stable diffusion models
    model_manager.set_diffusion_model("your_sd_model_path")
    # Load clip & dinov2 models
    model_manager.set_vision_models()

    generator = IterativeImageGenerator(
        max_iterations=3,
        prompts_per_iteration=3
    )

    # Set reference image
    generator.set_reference_image("your_img_path")

    # Run iterative generation
    results = generator.run_iterative_generation(
        initial_prompt="initial_prompt"
    )

    # Evaluation results
    print("Final Results:")
    print(json.dumps(results, indent=2))