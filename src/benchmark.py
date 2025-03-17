import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from iut import IUTGenerator, PromptGenerator
from iter import IterativeImageGenerator

from utils.model_manager import model_manager

class ImageRegenerationBenchmark:
    """
    Benchmark for evaluating T2I models on style/content datasets
    """
    
    def __init__(
        self,
        max_iterations: int = 3,
        prompts_per_iteration: int = 3,
        output_dir: str = "benchmark_results"
    ):
        """
        Initialize benchmark
        
        Args:
            model_name: Name of T2I model to evaluate
            openai_api_key: OpenAI API key
            max_iterations: Max iterations per image
            prompts_per_iteration: Number of prompt variants
            output_dir: Output directory
        """
        self.max_iterations = max_iterations
        self.prompts_per_iteration = prompts_per_iteration
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.iut_generator = IUTGenerator()
        self.iter_generator = IterativeImageGenerator(
            max_iterations=max_iterations,
            prompts_per_iteration=prompts_per_iteration
        )

    def run_single_case(self, image_path: str) -> Dict:
        """
        Run full pipeline for single image
        
        Args:
            image_path: Path to reference image
            
        Returns:
            Dictionary containing evaluation results
        """
        # Set reference image
        self.iut_generator.set_reference_image(image_path)
        self.iter_generator.set_reference_image(image_path)
        
        # Generate IUT
        iut = self.iut_generator.generate_iut()
        prompt_generator = PromptGenerator(
            iut_data=iut
        )
        iut_prompt = prompt_generator.run()
        print(iut_prompt)

        # Run iterative generation
        workspace_path = str(self.output_dir / f"{os.path.basename(image_path)}")
        self.iter_generator.set_workspace(workspace_path)
        iut_path = workspace_path + "/iut.json"
        self.iut_generator.save_iut(iut_path)
        results = self.iter_generator.run_iterative_generation(
            initial_prompt=iut_prompt
        )
        
        return {
            "image_path": str(image_path),
            "iut": iut,
            "results": results
        }

    def run_benchmark(self, dataset_path: str) -> Dict:
        """
        Run benchmark on entire dataset
        
        Args:
            dataset_path: Path to style/content dataset
            
        Returns:
            Dictionary containing aggregate results
        """
        image_paths = [str(p) for p in Path(dataset_path).glob("*.png")]
        if not image_paths:
            raise ValueError(f"No images found in {dataset_path}")
        
        # Run benchmark
        results = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                case_result = self.run_single_case(img_path)
                results.append(case_result)
                
                # Save individual results
                img_path = Path(img_path)
                case_dir = self.output_dir / img_path.stem
                os.makedirs(case_dir, exist_ok=True)
                with open(case_dir / "results.json", "w") as f:
                    json.dump(case_result, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Calculate aggregate metrics
        return self._calculate_aggregate_metrics(results)

    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate aggregate metrics from all results
        
        Args:
            results: List of individual case results
            
        Returns:
            Dictionary containing aggregate metrics
        """
        total = len(results)
        if total == 0:
            raise ValueError("No valid results to aggregate")
        
        # Initialize accumulators
        metrics = {
            "clip": 0,
            "dinov2": 0,
            "gpt4v_aesthetic": 0,
            "gpt4v_semantic": 0,
            "gpt4v_avg": 0
        }
        
        # Accumulate scores
        for result in results:
            final_scores = result["results"]
            metrics["clip"] += final_scores["clip_score"]
            metrics["dinov2"] += final_scores["dinov2_score"]
            metrics["gpt4v_aesthetic"] += int(final_scores["gpt4v_scores"]["aesthetic"])
            metrics["gpt4v_semantic"] += int(final_scores["gpt4v_scores"]["semantic"])
            metrics["gpt4v_avg"] += float(final_scores["gpt4v_scores"]["average"])
        
        # Calculate averages
        for key in metrics:
            metrics[key] /= total
        
        return {
            "total_images": total,
            "average_scores": metrics,
            "model_name": model_manager.sd_model_name
        }

    def save_benchmark_results(self, results: Dict, dataset_name: str):
        """
        Save benchmark results to file
        
        Args:
            results: Aggregate results dictionary
            dataset_name: Name of dataset (style/content)
        """
        output_path = self.output_dir / f"{dataset_name}_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

def main():

    
    # GPT model
    model_manager.set_llm_model(model_type="openai", openai_key="your_openai_api_key")
    # Qwen model
    # model_manager.set_llm_model(model_type="qwen", model_path='your_qwen_path')
    
    
    # Load stable diffusion models
    model_manager.set_diffusion_model("your_sd_model_path")
    # Load clip & dinov2 models
    model_manager.set_vision_models()


    # Initialize benchmark
    benchmark = ImageRegenerationBenchmark()
    
    # Run benchmark
    dataset_name = 'Evalref_100'
    dataset_path = '../data/Evalref_100'
    
    results = benchmark.run_benchmark(dataset_path)
    
    # Save and display results
    benchmark.save_benchmark_results(results, dataset_name)
    print("Benchmark Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()