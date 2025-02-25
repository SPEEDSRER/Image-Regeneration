import os
import json
import torch
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from io import BytesIO
import openai

from utils.model_manager import model_manager

class IterativeImageGenerator:
    """
    Iterative Image Generation with Multi-stage Processing
    """
    
    def __init__(
        self,
        model_type: str = "openai",
        sd_model_path: str = "",
        openai_api_key: str = None,
        max_iterations: int = 3,
        prompts_per_iteration: int = 3,
        output_dir: str = "results"
    ):
        # Hardware setup
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
        
        self.clip_model, self.clip_processor = model_manager.get_clip()
        self.dinov2_model, self.dinov2_processor = model_manager.get_dinov2()

        self.sd_model = model_manager.get_sd_model(sd_model_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.max_iterations = max_iterations
        self.prompts_per_iteration = prompts_per_iteration
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        
        # State variables
        self.ref_image = None
        self.ref_features = None
        self.current_prompt = None
        self.best_result = None

    
    def set_reference_image(self, image_path: str):
        """Set reference image and compute features"""
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
        if self.model_type == "openai": 
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate paraphrased prompts"},
                    {"role": "user", "content": f"Original: {base_prompt}\nGenerate {self.prompts_per_iteration} variants"}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return [choice.message.content for choice in response.choices[:self.prompts_per_iteration]]
        elif self.model_type == "qwen":
            query = self.qwen_tokenizer.from_list_format([
                {'text': f"Generate paraphrased prompts. Original: {base_prompt}\nGenerate {self.prompts_per_iteration} variants"}
            ])
            response, _ = self.qwen_model.chat(self.qwen_tokenizer, query=query, history=None)
            print(response)
            print([resp.strip() for resp in response.split('\n')[:self.prompts_per_iteration]])
            return [resp.strip() for resp in response.split('\n')[:self.prompts_per_iteration]]
            # return response
        else:
            raise ValueError("Unsupported model type")


    # Stage 2: Image Generation
    def _generate_images(self, prompts: List[str]) -> List[Tuple[str, Image.Image]]:
        """Generate images from prompts"""
        return [
            (prompt, self.pipe(prompt).images[0])
            for prompt in prompts
        ]

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
    def _generate_feedback(self, image: Image.Image) -> str:
        """Generate improvement feedback using GPT-4V"""
        encoded_image = self._encode_image(image)
        encoded_ref = self._encode_image(self.ref_image)
        if self.model_type == "openai":
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze differences and provide improvement suggestions:"},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_ref}"}
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        elif self.model_type == "qwen":
            response = "The images are nearly identical"
            print(response)
            return response
        else:
            raise ValueError("Unsupported model type")


    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
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
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_ref}"}
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
            print(variants)
            # Stage 2: Image Generation
            candidates = self._generate_images(variants)
            
            # Stage 3: Image Selection
            best_prompt, best_image, scores = self._select_best_candidate(candidates)
            
            # Stage 4: Feedback Generation
            feedback = self._generate_feedback(best_image)
            
            # Update state
            self.current_prompt = best_prompt
            history.append({
                "iteration": iter_num + 1,
                "prompt": best_prompt,
                "scores": scores,
                "feedback": feedback
            })
            
            # Save intermediate results
            self._save_iteration_results(iter_num, best_image, best_prompt, scores)

        # Final Evaluation
        final_scores = self._get_gpt4v_scores(best_image)
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
        iter_dir = self.output_dir / f"iter_{iteration+1}"
        os.makedirs(iter_dir, exist_ok=True)
        
        # Save image
        image.save(iter_dir / "best_image.jpg")
        
        # Save metadata
        with open(iter_dir / "metadata.json", "w") as f:
            json.dump({
                "prompt": prompt,
                "scores": scores
            }, f, indent=2)

    def _save_final_results(self, results: Dict, image: Image.Image):
        """Save final results"""
        final_dir = self.output_dir / "final"
        os.makedirs(final_dir, exist_ok=True)
        
        # Save image
        image.save(final_dir / "best_final.jpg")
        
        # Save metadata
        with open(final_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=2)



if __name__ == "__main__":
    # Initialize generator
    generator = IterativeImageGenerator(
        model_type="qwen",
        openai_api_key="your_api_key",
        sd_model_path='',
        max_iterations=3,
        prompts_per_iteration=3
    )

    # Set reference image
    generator.set_reference_image("test.jpg")

    # Run iterative generation
    results = generator.run_iterative_generation(
        initial_prompt="A girl is rowing a boat with a cat"
    )

    print("Final Results:")
    print(json.dumps(results, indent=2))