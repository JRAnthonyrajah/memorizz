# LongMemEval Evaluation for Memorizz

This directory contains the evaluation script for testing Memorizz's long-term memory capabilities using the LongMemEval benchmark.

## Setup

### 1. Download the Dataset

The LongMemEval dataset needs to be downloaded manually from the official repository:

```bash
# Run the download helper script
python download_dataset.py
```

This will provide instructions for downloading the dataset files. You need to:

1. Visit https://github.com/xiaowu0162/LongMemEval
2. Follow their setup instructions
3. Download the dataset files:
   - `longmemeval_oracle.json`
   - `longmemeval_s.json`
   - `longmemeval_m.json`
4. Place these files in the `data/` directory

### 2. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install datasets transformers
```

### 3. Configure Environment Variables

The script requires OpenAI API access for evaluation. Set your API key:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Optionally, configure MongoDB for memory storage:

```bash
export MONGODB_URI="your-mongodb-connection-string"
```

## Usage

### Basic Evaluation

Run the evaluation with default settings (oracle variant, 50 samples):

```bash
python evaluate_memorizz.py
```

### Custom Configuration

```bash
python evaluate_memorizz.py \
    --dataset_variant oracle \
    --num_samples 100 \
    --application_mode general \
    --output_dir ./results \
    --verbose
```

### Parameters

- `--dataset_variant`: Choose from "oracle", "s", or "m" (default: "oracle")
- `--num_samples`: Number of samples to evaluate (default: 50)
- `--application_mode`: Memorizz application mode to use (default: "general")
- `--output_dir`: Directory to save results (default: "./results")
- `--verbose`: Enable verbose logging

### Dataset Variants

- **oracle**: Contains only the evidence sessions (easier, for testing)
- **s**: Short version with ~40 history sessions (~115k tokens)
- **m**: Medium version with ~500 history sessions (much longer)

## Output

The evaluation script will:

1. Load the specified dataset variant
2. Create fresh Memorizz agents for each sample
3. Process conversation histories to build memory
4. Ask evaluation questions and collect responses
5. Use GPT-4 to evaluate response quality
6. Save detailed results to JSON files

Results include:
- Overall accuracy and scores
- Performance by category (IE, MR, KU, TR, ABS)
- Detailed per-sample results
- Processing time statistics

## Example Output

```
EVALUATION SUMMARY
==================================================
Dataset Variant: oracle
Application Mode: general
Samples Evaluated: 50
Overall Accuracy: 0.720
Overall Score: 0.756
Processing Time: 245.67s

Category Performance:
  information_extraction: 0.850 (12 samples)
  multi_session_reasoning: 0.667 (15 samples)
  knowledge_updates: 0.700 (10 samples)
  temporal_reasoning: 0.600 (8 samples)
  abstention: 0.800 (5 samples)

Detailed results saved to: ./results/longmemeval_oracle_general_20241201_143022.json
```

## Troubleshooting

### Dataset Not Found

If you get a "Dataset file not found" error:
1. Make sure you've downloaded the dataset files
2. Check that they're in the correct `data/` directory
3. Verify the filenames match exactly

### Memory Provider Issues

If MongoDB connection fails, the script will fall back to the default memory provider. For best results, configure a proper MongoDB instance.

### API Rate Limits

The evaluation uses GPT-4 for scoring, which may hit rate limits with large evaluations. Consider:
- Using smaller `num_samples` values
- Adding delays between API calls
- Using a higher-tier OpenAI account 