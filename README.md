# EDR Log Analysis with LLMs

This project demonstrates the use of Large Language Models (specifically Anthropic's Claude or Deepseek) to analyze Endpoint Detection and Response (EDR) logs for threat detection and investigation. It serves as a proof of concept to evaluate the capabilities and limitations of using LLMs for cybersecurity threat analysis.

## Overview

EDR logs contain a wealth of information about activities occurring on endpoints, but analyzing these logs for potential security threats requires domain expertise and can be time-consuming. This project explores whether LLMs can effectively analyze EDR logs to identify suspicious patterns and potential security threats, while highlighting the limitations of such an approach.

### Key Features

- **EDR Log Processing**: Load and preprocess EDR logs from Carbon Black or JSON/CSV files
- **Multiple LLM Support**: Works with Anthropic Claude or Deepseek models (cloud API or local inference)
- **Chunked Processing**: Handle large datasets by processing in manageable chunks
- **Attack Chain Identification**: Connect related threats to identify potential attack chains
- **Visualization**: Generate visualizations of analysis results
- **Comprehensive Reports**: Create detailed reports of findings with recommendations

### Primary Limitations Demonstrated

- **Context Window Constraints**: LLMs can only analyze a limited number of events at once
- **Probabilistic Correlation**: Reliance on pattern matching rather than deterministic rules
- **Resource Requirements**: High compute and API cost requirements that scale with data size
- **Lack of Memory**: Difficulty tracking complex attack patterns across analysis chunks

## Installation

### Prerequisites

- Python 3.8+
- Anthropic API key (for Claude)
- Carbon Black API credentials (optional, for direct CB access)
- For local Deepseek inference:
  - CUDA-capable GPU with 24GB+ VRAM (recommended)
  - At least 16GB system RAM

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/edr-llm-analyzer.git
   cd edr-llm-analyzer
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your API keys:
   - Create a copy of `config/config.yaml.example` as `config/config.yaml`
   - Add your API keys and other configuration options
   - For Anthropic Claude: set `llm.provider` to `"anthropic"` and add your API key
   - For Deepseek: set `llm.provider` to `"deepseek"` and configure the `deepseek` section

## Setting Up Local Deepseek Inference

TThis project demonstrates the resource limitations of using LLMs for security analysis by supporting local model inference. Follow these steps to run Deepseek models on your own hardware:

### Hardware Requirements

- **GPU Option:** CUDA-capable GPU with 24GB+ VRAM recommended (e.g., RTX 3090, 4090, A5000)
- **CPU Option:** Modern CPU with at least 16GB RAM (will be significantly slower)
- At least 50GB free disk space for model files

### Option 1: Using Deepseek with llama.cpp

For systems with limited GPU resources, you can run Deepseek using llama.cpp:

1. 1. **Install llama.cpp:**
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   
   # Build the project
   make
   
   # For CUDA support (if available)
   make LLAMA_CUBLAS=1
   ```

2. Download the GGUF version of Deepseek:
   ```bash
   # Create models directory
   mkdir -p models
   cd models

   # Download deepseek-coder 6.7B GGUF model (Q5_K_M is a good balance of quality and performance)
   wget https://huggingface.co/TheBloke/deepseek-coder-6.7B-base-GGUF/resolve/main/deepseek-coder-6.7b-base.Q5_K_M.gguf
   cd ..
   ```

3. Start the server:
   ```bash
   # Run the server on port 8080
   ./llama.cpp/server -m models/deepseek-coder-6.7b-base.Q5_K_M.gguf --host 0.0.0.0 --port 8080
   ```

4. Update your `config.yaml` to use the local llama.cpp server:
   ```yaml
   llm:
     provider: "deepseek_local"

   deepseek_local:
     model: "localhost"
     api_base: "http://localhost:8080/v1"
     measure_performance: true  # Track resource usage
   ```

### Option 2: Using Deepseek with vLLM

For systems with more powerful GPUs (24GB+ VRAM recommended):

1. Install vLLM:
   ```bash
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install vLLM
   pip install vllm
   ```

2. Download the Deepseek model from Hugging Face:
   ```bash
   # This will download the model (may take some time)
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='deepseek-ai/deepseek-coder-6.7b-base')"
   ```

3. Start the vLLM server:
   ```bash
      python -m vllm.entrypoints.openai.api_server \
     --model deepseek-ai/deepseek-coder-6.7b-base \
     --host 0.0.0.0 \
     --port 8000
   ```

4. Update your `config.yaml` to use the local vLLM server:
   ```yaml
   llm:
     provider: "deepseek_local"

   deepseek_local:
     model: "deepseek-ai/deepseek-coder-6.7b-base"
     api_base: "http://localhost:8000/v1"
     measure_performance: true  # Track resource usage
   ```

### Resource Measurements

When running with --measure-perf or setting measure_performance: true in the config, the tool will report:

- Peak memory usage during analysis
- GPU memory utilization (if available)
- Inference time per event and per chunk
- Total processing time
- CPU utilization during analysis

These measurements help demonstrate the resource limitations of using LLMs for security analysis.

### Troubleshooting Local Model Setup

1. **Connection Issues:** Ensure the server is running and the port in your config matches the port the server is running on
2. **Out of Memory Errors:** Try a smaller model or quantized version (e.g., switch to llama.cpp with a GGUF model)
3. **Slow Performance:** CPU inference will be much slower than GPU inference - consider using a cloud API if processing speed is important
4. **Model Loading Failures:** Ensure you have enough disk space and have downloaded the model correctly


## Usage

### Basic Usage

```bash
python src/main.py --input path/to/edr_logs --output path/to/output
```

### Command Line Options

```
--input, -i: Path to the input EDR log file or directory (required)
--output, -o: Path to the output directory (default: ./data/output)
--config, -c: Path to the configuration file (default: ./config/config.yaml)
--verbose, -v: Enable verbose logging
--no-api, -n: Run without making API calls (for testing)
--local-model, -l: Use local inference (requires local model setup)
--measure-perf, -p: Measure and report performance metrics
```

### Example Commands

1. Analyze a JSON file of Carbon Black events using Claude:
   ```bash
   python src/main.py --input data/sample/cb_events.json --output results --verbose
   ```

2. Analyze a JSON file using local Deepseek:
   ```bash
   python src/main.py --input data/sample/cb_events.json --output results --local-model --measure-perf
   ```

3. Generate sample data for testing:
   ```bash
   python -c "from src.utils import generate_sample_data; generate_sample_data('data/sample', 200)"
   ```

4. Test without making API calls:
   ```bash
   python src/main.py --input data/sample/sample_edr_data.json --output results --no-api
   ```

## Project Structure

```
edr-llm-analyzer/
│
├── README.md                 # Project overview and usage instructions
├── requirements.txt          # Dependencies
│
├── config/
│   └── config.yaml           # Configuration settings (API keys, model parameters)
│
├── data/
│   ├── sample/               # Sample EDR logs for testing
│   └── output/               # Analysis results
│
├── src/
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── data_loader.py        # Load and preprocess EDR logs
│   ├── llm_analyzer.py       # Interface with LLM APIs and local models
│   ├── visualizer.py         # Visualization of results
│   └── utils.py              # Helper functions
│
└── tests/                    # Unit tests
    ├── __init__.py
    ├── test_data_loader.py
    └── test_llm_analyzer.py
```


## Configuration

The `config.yaml` file contains settings for:

- **LLM Configuration**: Model selection, API parameters
- **EDR Data Configuration**: Log format, time window, event limits
- **Analysis Configuration**: Chunk size, overlap, timeouts
- **Visualization Configuration**: Graph formats, color schemes
- **Output Configuration**: Report format, verbosity

### Example Configurations

#### Using Anthropic Claude
```yaml
# LLM Configuration
llm:
  provider: "anthropic"
  model: "claude-3-7-sonnet-20250219"
  api_key: "YOUR_ANTHROPIC_API_KEY"
  max_tokens: 4096
  temperature: 0.1
```

#### Using Deepseek
```yaml
# LLM Configuration
llm:
  provider: "deepseek"
  api_key: "YOUR_DEEPSEEK_API_KEY"  # Optional, can also be in deepseek section
  max_tokens: 4096
  temperature: 0.1

# Deepseek Configuration
deepseek:
  model: "deepseek-chat"  # Or other Deepseek model
  api_key: "YOUR_DEEPSEEK_API_KEY"  # If not in llm section
  api_base: "https://api.deepseek.com/v1"  # Deepseek API endpoint
```

## Example Output

The analysis generates:

1. **Markdown Report**: Comprehensive findings including:
   - Executive summary
   - Identified threats with severity and confidence
   - Suspicious patterns
   - Attack chains and timeline
   - Visualizations
   - Recommendations

2. **Visualizations**:
   - Event frequency over time
   - Threat severity distribution
   - Attack timeline
   - Threat confidence levels
   - Event type distribution

## Limitations

This project demonstrates several key limitations of using LLMs for security analysis:

1. **Context Window Constraints**: The LLM can only analyze a limited number of events at once, potentially missing patterns that span across larger datasets.

2. **Probabilistic Correlation**: LLM-based analysis relies on probabilistic pattern matching rather than deterministic rule-based detection, which may lead to false positives or missed connections.

3. **Domain Knowledge**: LLM understanding of security context is limited compared to specialized security tools.

4. **Resource Requirements**: LLM-based analysis requires significant computational resources and API costs, scaling linearly with the size of the dataset.

5. **Lack of Memory**: The LLM has no persistent memory between chunks, making it difficult to track complex attack patterns that evolve over time.

## LLM Comparison

This project supports both Anthropic Claude and Deepseek models, allowing you to compare their effectiveness for EDR log analysis.

### Claude vs. Deepseek for Security Analysis

| Feature | Anthropic Claude | Deepseek |
|---------|------------------|----------|
| Context Window | Larger (up to 200K tokens in Claude 3 Opus) | Typically smaller |
| Cost | Higher per token | Lower per token |
| Security Domain Knowledge | Strong in recent models | Varies by model version |
| JSON Output Structure | Very reliable | Varies by model |
| Processing Speed | Fast but costlier | Can be more cost-efficient |
| Availability | API access requires approval | More open access |

### When to Use Each Model

- **Anthropic Claude**: Better for complex security scenarios, large datasets, and when high accuracy is critical
- **Deepseek**: Better for cost-efficiency, exploratory analysis, and when open-source solutions are preferred

The project allows you to easily switch between models by changing the configuration, enabling direct comparisons of their effectiveness on the same dataset.

## Educational Value

This project serves as an educational resource for:

- Security professionals exploring LLM applications
- Data scientists working with EDR data
- Developers learning about security log analysis
- Researchers studying LLM limitations in specialized domains

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Anthropic for the Claude API
- Carbon Black for the EDR log format

---

**Disclaimer**: This tool is for educational and research purposes only. The analysis provided by the LLM should not be solely relied upon for security decision-making in production environments. Always validate findings with traditional security tools and human expertise.