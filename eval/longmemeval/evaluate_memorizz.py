#!/usr/bin/env python3
"""
LongMemEval Evaluation Script for Memorizz

This script evaluates Memorizz's long-term memory capabilities using the LongMemEval benchmark.
It loads the dataset, creates Memorizz agents, processes conversations, and measures performance
across five core memory abilities.
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Place in environment variables
os.environ["MEMORIZZ_LOG_LEVEL"] = "WARNING"
os.environ["MONGODB_URI"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["VOYAGE_API_KEY"] = ""

try:
    pass
except ImportError:
    pass

try:
    # Try importing from installed package first, then fall back to local development
    try:
        print("Importing from installed package")
        from memorizz import MemAgent, MemoryProvider
        from memorizz.memory_provider.mongodb import MongoDBProvider, MongoDBConfig
        from memorizz.llms.openai import OpenAI
    except ImportError:
        print("Importing from local development")
        # Fall back to local development imports
        from src.memorizz import MemAgent, MemoryProvider
        from src.memorizz.memory_provider.mongodb.provider import MongoDBProvider, MongoDBConfig
        from src.memorizz.llms.openai import OpenAI
except ImportError as e:
    print(f"Error importing Memorizz: {e}")
    print("Make sure you're running from the project root and Memorizz is properly installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('longmemeval_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LongMemEvalEvaluator:
    """Evaluator for Memorizz using LongMemEval benchmark."""
    
    def __init__(self, 
                 dataset_variant: str = "oracle",
                 application_mode: str = "assistant",
                 output_dir: str = "./results",
                 verbose: bool = False):
        """
        Initialize the evaluator.
        
        Args:
            dataset_variant: LongMemEval variant ("oracle", "s", "m")
            application_mode: Memorizz application mode to use
            output_dir: Directory to save results
            verbose: Enable verbose logging
        """
        self.dataset_variant = dataset_variant
        self.application_mode = application_mode
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory provider
        self.memory_provider = self._init_memory_provider()
        
        # Initialize evaluation model for scoring
        self.eval_model = OpenAI(model="gpt-4")
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Category mapping - updated to match actual LongMemEval question types
        self.categories = {
            "single-session-user": "SSU",
            "single-session-assistant": "SSA", 
            "single-session-preference": "SSP",
            "multi-session": "MS",
            "temporal-reasoning": "TR",
            "knowledge-update": "KU"
        }
        
        logger.info(f"Initialized LongMemEval evaluator with variant: {dataset_variant}")
        
    def _init_memory_provider(self) -> MemoryProvider:
        """Initialize memory provider."""
        mongodb_uri = os.environ.get("MONGODB_URI")
        if not mongodb_uri:
            logger.warning("MONGODB_URI not found, using default memory provider")
            return MemoryProvider()
        
        try:
            config = MongoDBConfig(uri=mongodb_uri,
                db_name="testing_memorizz",
                embedding_provider="voyageai",
                embedding_config={
                    "model": "voyage-3-large",
                    "output_dimension": 256,
                })
            return MongoDBProvider(config)
        except Exception as e:
            logger.warning(f"Failed to initialize MongoDB provider: {e}, using default")
            return MemoryProvider()
    
    def _load_dataset(self):
        """Load LongMemEval dataset from local files."""
        try:
            # First, try to load from local data directory
            data_dir = Path(__file__).parent / "data"
            
            # Map dataset variants to filenames
            filename_map = {
                "oracle": "longmemeval_oracle.json",
                "s": "longmemeval_s.json", 
                "m": "longmemeval_m.json"
            }
            
            if self.dataset_variant not in filename_map:
                raise ValueError(f"Unknown dataset variant: {self.dataset_variant}")
            
            filename = filename_map[self.dataset_variant]
            filepath = data_dir / filename
            
            if filepath.exists():
                logger.info(f"Loading dataset from local file: {filepath}")
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Return the raw data directly - no need for HuggingFace datasets
                logger.info(f"Loaded LongMemEval-{self.dataset_variant.upper()} dataset with {len(data)} samples")
                return data
            else:
                # If local file doesn't exist, provide instructions for downloading
                logger.error(f"Dataset file not found: {filepath}")
                logger.error("Please download the LongMemEval dataset by running:")
                logger.error("python download_dataset.py")
                logger.error(f"This will download and extract the files to: {data_dir}")
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _create_fresh_agent(self) -> MemAgent:
        """Create a fresh Memorizz agent for evaluation."""
        print("Creating fresh agent memagent with specified memory provider and application mode")
        agent = MemAgent(
            memory_provider=self.memory_provider,
            application_mode=self.application_mode,
            instruction="You are a helpful assistant with excellent memory. Pay close attention to all conversations and remember important details about users and their preferences."
        )
        
        # Save the agent
        agent.save()
        return agent
    
    def _process_conversation_history(self, agent: MemAgent, history: List[List[Dict[str, Any]]]) -> None:
        """
        Process conversation history session by session to build up agent memory.
        
        Args:
            agent: The Memorizz agent
            history: List of conversation sessions, where each session is a list of messages
        """
        for session_idx, session in enumerate(history):
            if self.verbose:
                logger.info(f"Processing session {session_idx + 1}/{len(history)}")
            
            # Each session is directly a list of messages
            for message in session:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "user":
                    # Process user message through the agent to build memory
                    try:
                        response = agent.run(content)
                        if self.verbose:
                            logger.debug(f"User: {content[:100]}...")
                            logger.debug(f"Agent: {response[:100]}...")
                    except Exception as e:
                        logger.warning(f"Error processing message: {e}")
                        continue
    
    def _evaluate_response(self, question: str, agent_response: str, ground_truth: str, category: str) -> Dict[str, Any]:
        """
        Evaluate agent response using GPT-4 as a judge.
        
        Args:
            question: The evaluation question
            agent_response: Agent's response
            ground_truth: Expected answer
            category: Question category
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation_prompt = f"""
You are evaluating a chat assistant's response to a question about information from a long conversation history.

Question: {question}

Agent's Response: {agent_response}

Ground Truth Answer: {ground_truth}

Category: {category}

Please evaluate the agent's response on the following criteria:
1. Correctness: Does the response correctly answer the question?
2. Completeness: Does it provide sufficient detail?
3. Relevance: Is the response relevant to the question?

For categories involving abstention (when the agent should say "I don't know"), consider:
- If the ground truth indicates the information is unknown, the agent should abstain
- If the agent abstains when it should know the answer, that's incorrect
- If the agent provides an answer when it should abstain, that's incorrect

Provide your evaluation as a JSON object with:
{{
    "correct": true/false,
    "score": 0.0-1.0,
    "reasoning": "explanation of your evaluation"
}}

Only respond with the JSON object.
"""
        
        try:
            # Use the evaluation model - fix method name to generate_text
            eval_response = self.eval_model.generate_text(evaluation_prompt)
            
            # Parse JSON response
            eval_result = json.loads(eval_response)
            
            return {
                "correct": eval_result.get("correct", False),
                "score": eval_result.get("score", 0.0),
                "reasoning": eval_result.get("reasoning", ""),
                "agent_response": agent_response,
                "ground_truth": ground_truth
            }
            
        except Exception as e:
            logger.warning(f"Error in evaluation: {e}")
            return {
                "correct": False,
                "score": 0.0,
                "reasoning": f"Evaluation error: {e}",
                "agent_response": agent_response,
                "ground_truth": ground_truth
            }
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single sample from the dataset.
        
        Args:
            sample: A single evaluation sample
            
        Returns:
            Dictionary with evaluation results
        """
        start_time = time.time()
        
        # Extract sample information with correct field names
        question = sample["question"]
        ground_truth = sample["answer"]
        category = sample["question_type"]  # Changed from "category" to "question_type"
        
        # Handle haystack_sessions which might be a string that needs parsing
        history_raw = sample["haystack_sessions"]
        if isinstance(history_raw, str):
            try:
                import ast
                history = ast.literal_eval(history_raw)
            except:
                # If parsing fails, try json.loads
                try:
                    history = json.loads(history_raw)
                except:
                    logger.warning(f"Could not parse haystack_sessions: {history_raw[:100]}...")
                    history = []
        else:
            history = history_raw
        
        if self.verbose:
            logger.info(f"Evaluating question: {question[:100]}...")
            logger.info(f"Category: {category}")
        
        # Create fresh agent for this sample
        agent = self._create_fresh_agent()
        
        try:
            # Process conversation history
            self._process_conversation_history(agent, history)
            
            # Ask the evaluation question
            agent_response = agent.run(question)
            
            # Evaluate the response
            evaluation = self._evaluate_response(question, agent_response, ground_truth, category)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            result = {
                "question": question,
                "category": category,
                "agent_response": agent_response,
                "ground_truth": ground_truth,
                "evaluation": evaluation,
                "processing_time": processing_time,
                "history_length": len(history) if history else 0
            }
            
            # Clean up agent
            try:
                agent.delete(cascade=True)
            except Exception as e:
                logger.warning(f"Error cleaning up agent: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating sample: {e}")
            # Clean up agent on error
            try:
                agent.delete(cascade=True)
            except:
                pass
            
            return {
                "question": question,
                "category": category,
                "agent_response": f"Error: {e}",
                "ground_truth": ground_truth,
                "evaluation": {
                    "correct": False,
                    "score": 0.0,
                    "reasoning": f"Evaluation error: {e}"
                },
                "processing_time": time.time() - start_time,
                "history_length": len(history) if history else 0
            }
    
    def evaluate(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        Run evaluation on specified number of samples.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info(f"Starting evaluation on {num_samples} samples...")
        
        # Sample from dataset (self.dataset is now a list)
        total_samples = len(self.dataset)
        num_samples = min(num_samples, total_samples)
        samples = self.dataset[:num_samples]
        
        results = []
        category_scores = {cat: [] for cat in self.categories.keys()}
        
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}")
            
            result = self.evaluate_sample(sample)
            results.append(result)
            
            # Track category performance
            category = result["category"]
            score = result["evaluation"]["score"]
            
            if category in category_scores:
                category_scores[category].append(score)
        
        # Calculate aggregate metrics
        overall_scores = [r["evaluation"]["score"] for r in results]
        overall_accuracy = sum(r["evaluation"]["correct"] for r in results) / len(results)
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        category_results = {}
        for category, scores in category_scores.items():
            if scores:
                category_results[category] = {
                    "accuracy": sum(1 for s in scores if s >= 0.5) / len(scores),
                    "average_score": sum(scores) / len(scores),
                    "num_samples": len(scores)
                }
            else:
                category_results[category] = {
                    "accuracy": 0.0,
                    "average_score": 0.0,
                    "num_samples": 0
                }
        
        # Compile final results
        evaluation_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_variant": self.dataset_variant,
                "application_mode": self.application_mode,
                "num_samples": len(results),
                "total_processing_time": sum(r["processing_time"] for r in results)
            },
            "overall_accuracy": overall_accuracy,
            "overall_score": overall_score,
            "category_results": category_results,
            "detailed_results": results
        }
        
        return evaluation_results
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save evaluation results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"longmemeval_{self.dataset_variant}_{self.application_mode}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return filepath


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Memorizz using LongMemEval benchmark")
    
    parser.add_argument("--dataset_variant", choices=["oracle", "s", "m"], default="oracle",
                        help="LongMemEval dataset variant to use")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate")
    parser.add_argument("--application_mode", type=str, default="assistant",
                        help="Memorizz application mode to use")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()

    # Check for required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    try:
        # Initialize evaluator
        evaluator = LongMemEvalEvaluator(
            dataset_variant=args.dataset_variant,
            application_mode=args.application_mode,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Run evaluation
        results = evaluator.evaluate(num_samples=args.num_samples)
        
        # Save results
        output_file = evaluator.save_results(results)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Dataset Variant: {args.dataset_variant}")
        print(f"Application Mode: {args.application_mode}")
        print(f"Samples Evaluated: {results['metadata']['num_samples']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"Overall Score: {results['overall_score']:.3f}")
        print(f"Processing Time: {results['metadata']['total_processing_time']:.2f}s")
        print("\nCategory Performance:")
        for category, metrics in results['category_results'].items():
            print(f"  {category}: {metrics['accuracy']:.3f} ({metrics['num_samples']} samples)")
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 