<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  Chinese Documentation</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documentation</a>
<p>


> ⭐ If you like this project please click the "Star" button in the upper right corner to support us. Your support is our motivation to move forward!

## 📝 Introduction

EvalScope is a powerful and easily extensible model evaluation framework created by the [ModelScope Community](https://modelscope.cn/) aiming to provide a one-stop evaluation solution for large model developers.

Whether you want to evaluate the general capabilities of models conduct multi-model performance comparisons or need to stress test models EvalScope can meet your needs.

## ✨ Key Features

- **📚 Comprehensive Evaluation Benchmarks**: Built-in multiple industry-recognized evaluation benchmarks including MMLU C-Eval GSM8K and more.
- **🧩 Multi-modal and Multi-domain Support**: Supports evaluation of various model types including Large Language Models (LLM) Vision Language Models (VLM) Embedding Reranker AIGC and more.
- **🚀 Multi-backend Integration**: Seamlessly integrates multiple evaluation backends including OpenCompass VLMEvalKit RAGEval to meet different evaluation needs.
- **⚡ Inference Performance Testing**: Provides powerful model service stress testing tools supporting multiple performance metrics such as TTFT TPOT.
- **📊 Interactive Reports**: Provides WebUI visualization interface supporting multi-dimensional model comparison report overview and detailed inspection.
- **⚔️ Arena Mode**: Supports multi-model battles (Pairwise Battle) intuitively ranking and evaluating models.
- **🔧 Highly Extensible**: Developers can easily add custom datasets models and evaluation metrics.

<details><summary>🏛️ Overall Architecture</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope Overall Architecture.
</p>

1.  **Input Layer**
    - **Model Sources**: API models (OpenAI API) Local models (ModelScope)
    - **Datasets**: Standard evaluation benchmarks (MMLU/GSM8k etc.) Custom data (MCQ/QA)

2.  **Core Functions**
    - **Multi-backend Evaluation**: Native backend OpenCompass MTEB VLMEvalKit RAGAS
    - **Performance Monitoring**: Supports multiple model service APIs and data formats tracking TTFT/TPOP and other metrics
    - **Tool Extensions**: Integrates Tool-Bench Needle-in-a-Haystack etc.

3.  **Output Layer**
    - **Structured Reports**: Supports JSON Table Logs
    - **Visualization Platform**: Supports Gradio Wandb SwanLab

</details>

## 🎉 What's New

> [!IMPORTANT]
> **Version 1.0 Refactoring**
>
> Version 1.0 introduces a major overhaul of the evaluation framework establishing a new more modular and extensible API layer under `evalscope/api`. Key improvements include standardized data models for benchmarks samples and results; a registry-based design for components such as benchmarks and metrics; and a rewritten core evaluator that orchestrates the new architecture. Existing benchmark adapters have been migrated to this API resulting in cleaner more consistent and easier-to-maintain implementations.

- 🔥 **[2025.12.02]** Added support for custom multimodal VQA evaluation; refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html). Added support for visualizing model service stress testing in ClearML; refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#clearml).
- 🔥 **[2025.11.26]** Added support for OpenAI-MRCR GSM8K-V MGSM MicroVQA IFBench SciCode benchmarks.
- 🔥 **[2025.11.18]** Added support for custom Function-Call (tool invocation) datasets to test whether models can timely and correctly call tools. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#function-calling-format-fc).
- 🔥 **[2025.11.14]** Added support for SWE-bench_Verified SWE-bench_Lite SWE-bench_Verified_mini code evaluation benchmarks. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html).
- 🔥 **[2025.11.12]** Added `pass@k` `vote@k` `pass^k` and other metric aggregation methods; added support for multimodal evaluation benchmarks such as A_OKVQA CMMU ScienceQA V*Bench.
- 🔥 **[2025.11.07]** Added support for τ²-bench an extended and enhanced version of τ-bench that includes a series of code fixes and adds telecom domain troubleshooting scenarios. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/tau2_bench.html).
- 🔥 **[2025.10.30]** Added support for BFCL-v4 enabling evaluation of agent capabilities including web search and long-term memory. See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v4.html).
- 🔥 **[2025.10.27]** Added support for LogiQA HaluEval MathQA MRI-QA PIQA QASC CommonsenseQA and other evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.26]** Added support for Conll-2003 CrossNER Copious GeniaNER HarveyNER MIT-Movie-Trivia MIT-Restaurant OntoNotes5 WNUT2017 and other Named Entity Recognition evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.21]** Optimized sandbox environment usage in code evaluation supporting both local and remote operation modes. For details refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html).
- 🔥 **[2025.10.20]** Added support for evaluation benchmarks including PolyMath SimpleVQA MathVerse MathVision AA-LCR; optimized evalscope perf performance to align with vLLM Bench. For details refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/vs_vllm_bench.html).
- 🔥 **[2025.10.14]** Added support for OCRBench OCRBench-v2 DocVQA InfoVQA ChartQA and BLINK multimodal image-text evaluation benchmarks.
- 🔥 **[2025.09.22]** Code evaluation benchmarks (HumanEval LiveCodeBench) now support running in a sandbox environment. To use this feature please install [ms-enclave](https://github.com/modelscope/ms-enclave) first.
- 🔥 **[2025.09.19]** Added support for multimodal image-text evaluation benchmarks including RealWorldQA AI2D MMStar MMBench and OmniBench as well as pure text evaluation benchmarks such as Multi-IF HealthBench and AMC.
- 🔥 **[2025.09.05]** Added support for vision-language multimodal model evaluation tasks such as MathVista and MMMU. For more supported datasets please [refer to the documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/vlm.html).
- 🔥 **[2025.09.04]** Added support for image editing task evaluation including the [GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) benchmark. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/image_edit.html).
- 🔥 **[2025.08.22]** Version 1.0 Refactoring. Break changes please [refer to](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#switching-to-version-v1-0).
<details><summary>More</summary>

- 🔥 **[2025.07.18]** The model stress testing now supports randomly generating image-text data for multimodal model evaluation. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#id4).
- 🔥 **[2025.07.16]** Support for [τ-bench](https://github.com/sierra-research/tau-bench) has been added enabling the evaluation of AI Agent performance and reliability in real-world scenarios involving dynamic user and tool interactions. For usage instructions please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html#bench).
- 🔥 **[2025.07.14]** Support for "Humanity's Last Exam" ([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle)) a highly challenging evaluation benchmark. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam).
- 🔥 **[2025.07.03]** Refactored Arena Mode: now supports custom model battles outputs a model leaderboard and provides battle result visualization. See [reference](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html) for details.
- 🔥 **[2025.06.28]** Optimized custom dataset evaluation: now supports evaluation without reference answers. Enhanced LLM judge usage with built-in modes for "scoring directly without reference answers" and "checking answer consistency with reference answers". See [reference](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#qa) for details.
- 🔥 **[2025.06.19]** Added support for the [BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3) benchmark designed to evaluate model function-calling capabilities across various scenarios. For more information refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html).
- 🔥 **[2025.06.02]** Added support for the Needle-in-a-Haystack test. Simply specify `needle_haystack` to conduct the test and a corresponding heatmap will be generated in the `outputs/reports` folder providing a visual representation of the model's performance. Refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/needle_haystack.html) for more details.
- 🔥 **[2025.05.29]** Added support for two long document evaluation benchmarks: [DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary) and [FRAMES](https://modelscope.cn/datasets/iic/frames/summary). For usage guidelines please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html).
- 🔥 **[2025.05.16]** Model service performance stress testing now supports setting various levels of concurrency and outputs a performance test report. [Reference example](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html#id3).
- 🔥 **[2025.05.13]** Added support for the [ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static) dataset to evaluate model's tool-calling capabilities. Refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html) for usage instructions. Also added support for the [DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview) and [Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val) benchmarks to assess the reasoning capabilities of models.
- 🔥 **[2025.04.29]** Added Qwen3 Evaluation Best Practices [welcome to read 📖](https://evalscope.readthedocs.io/en/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** Support for text-to-image evaluation: Supports 8 metrics including MPS HPSv2.1Score etc. and evaluation benchmarks such as EvalMuse GenAI-Bench. Refer to the [user documentation](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/t2i.html) for more details.
- 🔥 **[2025.04.10]** Model service stress testing tool now supports the `/v1/completions` endpoint (the default endpoint for vLLM benchmarking)
- 🔥 **[2025.04.08]** Support for evaluating embedding model services compatible with the OpenAI API has been added. For more details check the [user guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters).
- 🔥 **[2025.03.27]** Added support for [AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview) and [ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary) evaluation benchmarks. For usage notes please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** The model inference service stress testing now supports generating prompts of specified length using random values. Refer to the [user guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#using-the-random-dataset) for more details.
- 🔥 **[2025.03.13]** Added support for the [LiveCodeBench](https://www.modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary) code evaluation benchmark which can be used by specifying `live_code_bench`. Supports evaluating QwQ-32B on LiveCodeBench refer to the [best practices](https://evalscope.readthedocs.io/en/latest/best_practice/eval_qwq.html).
- 🔥 **[2025.03.11]** Added support for the [SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary) and [Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary) evaluation benchmarks. These are used to assess the factual accuracy of models and you can specify `simple_qa` and `chinese_simpleqa` for use. Support for specifying a judge model is also available. For more details refer to the [relevant parameter documentation](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).
- 🔥 **[2025.03.07]** Added support for the [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/summary) model evaluate the model's reasoning ability and reasoning efficiency refer to [📖 Best Practices for QwQ-32B Evaluation](https://evalscope.readthedocs.io/en/latest/best_practice/eval_qwq.html) for more details.
- 🔥 **[2025.03.04]** Added support for the [SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary) dataset which covers 13 categories 72 first-level disciplines and 285 second-level disciplines totaling 26529 questions. You can use it by specifying `super_gpqa`.
- 🔥 **[2025.03.03]** Added support for evaluating the IQ and EQ of models. Refer to [📖 Best Practices for IQ and EQ Evaluation](https://evalscope.readthedocs.io/en/latest/best_practice/iquiz.html) to find out how smart your AI is!
- 🔥 **[2025.02.27]** Added support for evaluating the reasoning efficiency of models. Refer to [📖 Best Practices for Evaluating Thinking Efficiency](https://evalscope.readthedocs.io/en/latest/best_practice/think_eval.html). This implementation is inspired by the works [Overthinking](https://doi.org/10.48550/arXiv.2412.21187) and [Underthinking](https://doi.org/10.48550/arXiv.2501.18585).
- 🔥 **[2025.02.25]** Added support for two model inference-related evaluation benchmarks: [MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR) and [ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary). To use them simply specify `musr` and `process_bench` respectively in the datasets parameter.
- 🔥 **[2025.02.18]** Supports the AIME25 dataset which contains 15 questions (Grok3 scored 93 on this dataset).
- 🔥 **[2025.02.13]** Added support for evaluating DeepSeek distilled models including AIME24 MATH-500 and GPQA-Diamond datasets，refer to [best practice](https://evalscope.readthedocs.io/en/latest/best_practice/deepseek_r1_distill.html); Added support for specifying the `eval_batch_size` parameter to accelerate model evaluation.
- 🔥 **[2025.01.20]** Support for visualizing evaluation results including single model evaluation results and multi-model comparison refer to the [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html) for more details; Added [`iquiz`](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary) evaluation example evaluating the IQ and EQ of the model.
- 🔥 **[2025.01.07]** Native backend: Support for model API evaluation is now available. Refer to the [📖 Model API Evaluation Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#api) for more details. Additionally support for the `ifeval` evaluation benchmark has been added.
- 🔥🔥 **[2024.12.31]** Support for adding benchmark evaluations refer to the [📖 Benchmark Evaluation Addition Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html); support for custom mixed dataset evaluations allowing for more comprehensive model evaluations with less data refer to the [📖 Mixed Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/collection/index.html).
- 🔥 **[2024.12.13]** Model evaluation optimization: no need to pass the `--template-type` parameter anymore; supports starting evaluation with `evalscope eval --args`. Refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html) for more details.
- 🔥 **[2024.11.26]** The model inference service performance evaluator has been completely refactored: it now supports local inference service startup and Speed Benchmark; asynchronous call error handling has been optimized. For more details refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).
- 🔥 **[2024.10.31]** The best practice for evaluating Multimodal-RAG has been updated please check the [📖 Blog](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag) for more details.
- 🔥 **[2024.10.23]** Supports multimodal RAG evaluation including the assessment of image-text retrieval using [CLIP_Benchmark](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/clip_benchmark.html) and extends [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html) to support end-to-end multimodal metrics evaluation.
- 🔥 **[2024.10.8]** Support for RAG evaluation including independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html) as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html).
- 🔥 **[2024.09.18]** Our documentation has been updated to include a blog module featuring some technical research and discussions related to evaluations. We invite you to [📖 read it](https://evalscope.readthedocs.io/en/refact_readme/blog/index.html).
- 🔥 **[2024.09.12]** Support for LongWriter evaluation which supports 10000+ word generation. You can use the benchmark [LongBench-Write](evalscope/third_party/longbench_write/README.md) to measure the long output quality as well as the output length.
- 🔥 **[2024.08.30]** Support for custom dataset evaluations including text datasets and multimodal image-text datasets.
- 🔥 **[2024.08.20]** Updated the official documentation including getting started guides best practices and FAQs. Feel free to [📖read it here](https://evalscope.readthedocs.io/en/latest/)!
- 🔥 **[2024.08.09]** Simplified the installation process allowing for pypi installation of vlmeval dependencies; optimized the multimodal model evaluation experience achieving up to 10x acceleration based on the OpenAI API evaluation chain.
- 🔥 **[2024.07.31]** Important change: The package name `llmuses` has been changed to `evalscope`. Please update your code accordingly.
- 🔥 **[2024.07.26]** Support for **VLMEvalKit** as a third-party evaluation framework to initiate multimodal model evaluation tasks.
- 🔥 **[2024.06.29]** Support for **OpenCompass** as a third-party evaluation framework which we have encapsulated at a higher level supporting pip installation and simplifying evaluation task configuration.
- 🔥 **[2024.06.13]** EvalScope seamlessly integrates with the fine-tuning framework SWIFT providing full-chain support from LLM training to evaluation.
- 🔥 **[2024.06.13]** Integrated the Agent evaluation dataset ToolBench.

</details>

## ❤️ Community & Support

Welcome to join our community to communicate with other developers and get help.

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  WeChat Group | DingTalk Group
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">



## 🛠️ Environment Setup

We recommend using `conda` to create a virtual environment and install with `pip`.

1.  **Create and Activate Conda Environment** (Python 3.10 recommended)
    ```shell
    conda create -n evalscope python=3.10
    conda activate evalscope
    ```

2.  **Install EvalScope**

    - **Method 1: Install via PyPI (Recommended)**
      ```shell
      pip install evalscope
      ```

    - **Method 2: Install from Source (For Development)**
      ```shell
      git clone https://github.com/modelscope/evalscope.git
      cd evalscope
      pip install -e .
      ```

3.  **Install Additional Dependencies** (Optional)
    Install corresponding feature extensions according to your needs:
    ```shell
    # Performance testing
    pip install 'evalscope[perf]'

    # Visualization App
    pip install 'evalscope[app]'

    # Other evaluation backends
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'

    # Install all dependencies
    pip install 'evalscope[all]'
    ```
    > If you installed from source please replace `evalscope` with `.` for example `pip install '.[perf]'`.

> [!NOTE]
> This project was formerly known as `llmuses`. If you need to use `v0.4.3` or earlier versions please run `pip install llmuses<=0.4.3` and use `from llmuses import ...` for imports.


## 🚀 Quick Start

You can start evaluation tasks in two ways: **command line** or **Python code**.

### Method 1. Using Command Line

Execute the `evalscope eval` command in any path to start evaluation. The following command will evaluate the `Qwen/Qwen2.5-0.5B-Instruct` model on `gsm8k` and `arc` datasets taking only 5 samples from each dataset.

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### Method 2. Using Python Code

Use the `run_task` function and `TaskConfig` object to configure and start evaluation tasks.

```python
from evalscope import run_task TaskConfig

# Configure evaluation task
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct'
    datasets=['gsm8k' 'arc']
    limit=5
)

# Start evaluation
run_task(task_cfg)
```

<details><summary><b>💡 Tip:</b> `run_task` also supports dictionaries YAML or JSON files as configuration.</summary>

**Using Python Dictionary**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct'
    'datasets': ['gsm8k' 'arc']
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**Using YAML File** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### Output Results
After evaluation completion you will see a report in the terminal in the following format:
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 📈 Advanced Usage

### Custom Evaluation Parameters

You can fine-tune model loading inference and dataset configuration through command line parameters.

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master" "precision": "torch.float16" "device_map": "auto"}' \
 --generation-config '{"do_sample":true"temperature":0.6"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0 "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: Model loading parameters such as `revision` `precision` etc.
- `--generation-config`: Model generation parameters such as `temperature` `max_tokens` etc.
- `--dataset-args`: Dataset configuration parameters such as `few_shot_num` etc.

For details please refer to [📖 Complete Parameter Guide](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).

### Evaluating Online Model APIs

EvalScope supports evaluating model services deployed via APIs (such as services deployed with vLLM). Simply specify the service address and API Key.

1.  **Start Model Service** (using vLLM as example)
    ```shell
    export VLLM_USE_MODELSCOPE=True
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen2.5 \
      --port 8801
    ```

2.  **Run Evaluation**
    ```shell
    evalscope eval \
     --model qwen2.5 \
     --eval-type openai_api \
     --api-url http://127.0.0.1:8801/v1 \
     --api-key EMPTY \
     --datasets gsm8k \
     --limit 10
    ```

### ⚔️ Arena Mode

Arena mode evaluates model performance through pairwise battles between models providing win rates and rankings perfect for horizontal comparison of multiple models.

```text
# Example evaluation results
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
For details please refer to [📖 Arena Mode Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html).

### 🖊️ Custom Dataset Evaluation

EvalScope allows you to easily add and evaluate your own datasets. For details please refer to [📖 Custom Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html).


## 🧪 Other Evaluation Backends
EvalScope supports launching evaluation tasks through third-party evaluation frameworks (we call them "backends") to meet diverse evaluation needs.

- **Native**: EvalScope's default evaluation framework with comprehensive functionality.
- **OpenCompass**: Focuses on text-only evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: Focuses on multi-modal evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: Focuses on RAG evaluation supporting Embedding and Reranker models. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/index.html)
- **Third-party Evaluation Tools**: Supports evaluation tasks like [ToolBench](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html).

## ⚡ Inference Performance Evaluation Tool
EvalScope provides a powerful stress testing tool for evaluating the performance of large language model services.

- **Key Metrics**: Supports throughput (Tokens/s) first token latency (TTFT) token generation latency (TPOT) etc.
- **Result Recording**: Supports recording results to `wandb` and `swanlab`.
- **Speed Benchmarks**: Can generate speed benchmark results similar to official reports.

For details please refer to [📖 Performance Testing Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).

Example output is shown below:
<p align="center">
    <img src="docs/en/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>


## 📊 Visualizing Evaluation Results

EvalScope provides a Gradio-based WebUI for interactive analysis and comparison of evaluation results.

1.  **Install Dependencies**
    ```bash
    pip install 'evalscope[app]'
    ```

2.  **Start Service**
    ```bash
    evalscope app
    ```
    Visit `http://127.0.0.1:7861` to open the visualization interface.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/setting.png" alt="Setting" style="width: 85%;" />
      <p>Settings Interface</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/model_compare.png" alt="Model Compare" style="width: 100%;" />
      <p>Model Comparison</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/report_overview.png" alt="Report Overview" style="width: 100%;" />
      <p>Report Overview</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/report_details.png" alt="Report Details" style="width: 85%;" />
      <p>Report Details</p>
    </td>
  </tr>
</table>

For details please refer to [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html).

## 👷‍♂️ Contributing

We welcome any contributions from the community! If you want to add new evaluation benchmarks models or features please refer to our [Contributing Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html).

Thanks to all developers who have contributed to EvalScope!

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>


## 📚 Citation

If you use EvalScope in your research please cite our work:
```bibtex
@misc{evalscope_2024
    title={{EvalScope}: Evaluation Framework for Large Models}
    author={ModelScope Team}
    year={2024}
    url={https://github.com/modelscope/evalscope}
}
```


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)

<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  中文 &nbsp ｜ &nbsp <a href="evalscope.md">English</a> &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/zh-cn/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documents</a>
<p>


> ⭐ 如果你喜欢这个项目，请点击右上角的 "Star" 按钮支持我们。你的支持是我们前进的动力！

## 📝 简介

EvalScope 是由[魔搭社区](https://modelscope.cn/)打造的一款功能强大、易于扩展的模型评测框架，旨在为大模型开发者提供一站式评测解决方案。

无论您是想评估模型的通用能力、进行多模型性能对比，还是需要对模型进行压力测试，EvalScope 都能满足您的需求。

## ✨ 主要特性

- **📚 全面的评测基准**: 内置 MMLU C-Eval GSM8K 等多个业界公认的评测基准。
- **🧩 多模态与多领域支持**: 支持大语言模型 (LLM)、多模态 (VLM)、Embedding、Reranker、AIGC 等多种模型的评测。
- **🚀 多后端集成**: 无缝集成 OpenCompass VLMEvalKit RAGEval 等多种评测后端，满足不同评测需求。
- **⚡ 推理性能测试**: 提供强大的模型服务压力测试工具，支持 TTFT TPOT 等多项性能指标。
- **📊 交互式报告**: 提供 WebUI 可视化界面，支持多维度模型对比、报告概览和详情查阅。
- **⚔️ 竞技场模式**: 支持多模型对战 (Pairwise Battle)，直观地对模型进行排名和评估。
- **🔧 高度可扩展**: 开发者可以轻松添加自定义数据集、模型和评测指标。

<details><summary>🏛️ 整体架构</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope 整体架构图.
</p>

1.  **输入层**
    - **模型来源**: API模型（OpenAI API）、本地模型（ModelScope）
    - **数据集**: 标准评测基准（MMLU/GSM8k等）、自定义数据（MCQ/QA）

2.  **核心功能**
    - **多后端评估**: 原生后端、OpenCompass、MTEB、VLMEvalKit、RAGAS
    - **性能监控**: 支持多种模型服务 API 和数据格式，追踪 TTFT/TPOP 等指标
    - **工具扩展**: 集成 Tool-Bench Needle-in-a-Haystack 等

3.  **输出层**
    - **结构化报告**: 支持 JSON Table Logs
    - **可视化平台**: 支持 Gradio Wandb SwanLab

</details>

## 🎉 内容更新

> [!IMPORTANT]
> **版本 1.0 重构**
>
> 版本 1.0 对评测框架进行了重大重构，在 `evalscope/api` 下建立了全新的、更模块化且易扩展的 API 层。主要改进包括：为基准、样本和结果引入了标准化数据模型；对基准和指标等组件采用注册表式设计；并重写了核心评测器以协同新架构。现有的基准已迁移到这一 API，实现更加简洁、一致且易于维护。

- 🔥 **[2025.12.02]** 支持自定义多模态VQA评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html) ；支持模型服务压测在 ClearML 上可视化，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#clearml)。
- 🔥 **[2025.11.26]** 新增支持 OpenAI-MRCR、GSM8K-V、MGSM、MicroVQA、IFBench、SciCode 评测基准。
- 🔥 **[2025.11.18]** 支持自定义 Function-Call（工具调用）数据集，来测试模型能否适时并正确调用工具，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#fc)
- 🔥 **[2025.11.14]** 新增支持SWE-bench_Verified SWE-bench_Lite SWE-bench_Verified_mini 代码评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)。
- 🔥 **[2025.11.12]** 新增`pass@k`、`vote@k`、`pass^k`等指标聚合方法；新增支持A_OKVQA CMMU ScienceQ V*Bench等多模态评测基准。
- 🔥 **[2025.11.07]** 新增支持τ²-bench，是 τ-bench 的扩展与增强版本，包含一系列代码修复，并新增了电信（telecom）领域的故障排查场景，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau2_bench.html)。
- 🔥 **[2025.10.30]** 新增支持BFCL-v4，支持agent的网络搜索和长期记忆能力的评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v4.html)。
- 🔥 **[2025.10.27]** 新增支持LogiQA HaluEval MathQA MRI-QA PIQA QASC CommonsenseQA等评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.26]** 新增支持Conll-2003 CrossNER Copious GeniaNER HarveyNER MIT-Movie-Trivia MIT-Restaurant OntoNotes5 WNUT2017 等命名实体识别评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.21]** 优化代码评测中的沙箱环境使用，支持在本地和远程两种模式下运行，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 🔥 **[2025.10.20]** 新增支持PolyMath SimpleVQA MathVerse MathVision AA-LCR 等评测基准；优化evalscope perf表现，对齐vLLM Bench，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/vs_vllm_bench.html)。
- 🔥 **[2025.10.14]** 新增支持OCRBench OCRBench-v2 DocVQA InfoVQA ChartQA BLINK 等图文多模态评测基准。
- 🔥 **[2025.09.22]** 代码评测基准(HumanEval LiveCodeBench)支持在沙箱环境中运行，要使用该功能需先安装[ms-enclave](https://github.com/modelscope/ms-enclave)。
- 🔥 **[2025.09.19]** 新增支持RealWorldQA、AI2D、MMStar、MMBench、OmniBench等图文多模态评测基准，和Multi-IF、HealthBench、AMC等纯文本评测基准。
- 🔥 **[2025.09.05]** 支持视觉-语言多模态大模型的评测任务，例如：MathVista、MMMU，更多支持数据集请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/vlm.html)。
- 🔥 **[2025.09.04]** 支持图像编辑任务评测，支持[GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) 评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/image_edit.html)。
- 🔥 **[2025.08.22]** Version 1.0 重构，不兼容的更新请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#v1-0)。
<details> <summary>更多</summary>

- 🔥 **[2025.07.18]** 模型压测支持随机生成图文数据，用于多模态模型压测，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#id4)。
- 🔥 **[2025.07.16]** 支持[τ-bench](https://github.com/sierra-research/tau-bench)，用于评估 AI Agent在动态用户和工具交互的实际环境中的性能和可靠性，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#bench)。
- 🔥 **[2025.07.14]** 支持“人类最后的考试”([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle))，这一高难度评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam)。
- 🔥 **[2025.07.03]** 重构了竞技场模式，支持自定义模型对战，输出模型排行榜，以及对战结果可视化，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。
- 🔥 **[2025.06.28]** 优化自定义数据集评测，支持无参考答案评测；优化LLM裁判使用，预置“无参考答案直接打分” 和 “判断答案是否与参考答案一致”两种模式，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa)
- 🔥 **[2025.06.19]** 新增支持[BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3)评测基准，用于评测模型在多种场景下的函数调用能力，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)。
- 🔥 **[2025.06.02]** 新增支持大海捞针测试（Needle-in-a-Haystack），指定`needle_haystack`即可进行测试，并在`outputs/reports`文件夹下生成对应的heatmap，直观展现模型性能，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)。
- 🔥 **[2025.05.29]** 新增支持[DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)和[FRAMES](https://modelscope.cn/datasets/iic/frames/summary)两个长文档评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.05.16]** 模型服务性能压测支持设置多种并发，并输出性能压测报告，[参考示例](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id3)。
- 🔥 **[2025.05.13]** 新增支持[ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static)数据集，评测模型的工具调用能力，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)；支持[DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview)和[Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val)评测基准，评测模型的推理能力。
- 🔥 **[2025.04.29]** 新增Qwen3评测最佳实践，[欢迎阅读📖](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** 支持文生图评测：支持MPS、HPSv2.1Score等8个指标，支持EvalMuse、GenAI-Bench等评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/t2i.html)
- 🔥 **[2025.04.10]** 模型服务压测工具支持`/v1/completions`端点（也是vLLM基准测试的默认端点）
- 🔥 **[2025.04.08]** 支持OpenAI API兼容的Embedding模型服务评测，查看[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters)
- 🔥 **[2025.03.27]** 新增支持[AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview)和[ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** 模型推理服务压测支持random生成指定范围长度的prompt，参考[使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random)
- 🔥 **[2025.03.13]** 新增支持[LiveCodeBench](https://www.modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary)代码评测基准，指定`live_code_bench`即可使用；支持QwQ-32B 在LiveCodeBench上评测，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.11]** 新增支持[SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)和[Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)评测基准，用与评测模型的事实正确性，指定`simple_qa`和`chinese_simpleqa`使用。同时支持指定裁判模型，参考[相关参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。
- 🔥 **[2025.03.07]** 新增QwQ-32B模型评测最佳实践，评测了模型的推理能力以及推理效率，参考[📖QwQ-32B模型评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.04]** 新增支持[SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)数据集，其覆盖 13 个门类、72 个一级学科和 285 个二级学科，共 26529 个问题，指定`super_gpqa`即可使用。
- 🔥 **[2025.03.03]** 新增支持评测模型的智商和情商，参考[📖智商和情商评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/iquiz.html)，来测测你家的AI有多聪明？
- 🔥 **[2025.02.27]** 新增支持评测推理模型的思考效率，参考[📖思考效率评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/think_eval.html)，该实现参考了[Overthinking](https://doi.org/10.48550/arXiv.2412.21187) 和 [Underthinking](https://doi.org/10.48550/arXiv.2501.18585)两篇工作。
- 🔥 **[2025.02.25]** 新增支持[MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR)和[ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary)两个模型推理相关评测基准，datasets分别指定`musr`和`process_bench`即可使用。
- 🔥 **[2025.02.18]** 支持AIME25数据集，包含15道题目（Grok3 在该数据集上得分为93分）
- 🔥 **[2025.02.13]** 支持DeepSeek蒸馏模型评测，包括AIME24 MATH-500 GPQA-Diamond数据集，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/deepseek_r1_distill.html)；支持指定`eval_batch_size`参数，加速模型评测
- 🔥 **[2025.01.20]** 支持可视化评测结果，包括单模型评测结果和多模型评测结果对比，参考[📖可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)；新增[`iquiz`](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)评测样例，评测模型的IQ和EQ。
- 🔥 **[2025.01.07]** Native backend: 支持模型API评测，参考[📖模型API评测指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#api)；新增支持`ifeval`评测基准。
- 🔥🔥 **[2024.12.31]** 支持基准评测添加，参考[📖基准评测添加指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)；支持自定义混合数据集评测，用更少的数据，更全面的评测模型，参考[📖混合数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html)
- 🔥 **[2024.12.13]** 模型评测优化，不再需要传递`--template-type`参数；支持`evalscope eval --args`启动评测，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html)
- 🔥 **[2024.11.26]** 模型推理压测工具重构完成：支持本地启动推理服务、支持Speed Benchmark；优化异步调用错误处理，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)
- 🔥 **[2024.10.31]** 多模态RAG评测最佳实践发布，参考[📖博客](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag)
- 🔥 **[2024.10.23]** 支持多模态RAG评测，包括[CLIP_Benchmark](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/clip_benchmark.html)评测图文检索器，以及扩展了[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)以支持端到端多模态指标评测。
- 🔥 **[2024.10.8]** 支持RAG评测，包括使用[MTEB/CMTEB](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html)进行embedding模型和reranker的独立评测，以及使用[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)进行端到端评测。
- 🔥 **[2024.09.18]** 我们的文档增加了博客模块，包含一些评测相关的技术调研和分享，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/blog/index.html)
- 🔥 **[2024.09.12]** 支持 LongWriter 评测，您可以使用基准测试 [LongBench-Write](evalscope/third_party/longbench_write/README.md) 来评测长输出的质量以及输出长度。
- 🔥 **[2024.08.30]** 支持自定义数据集评测，包括文本数据集和多模态图文数据集。
- 🔥 **[2024.08.20]** 更新了官方文档，包括快速上手、最佳实践和常见问题等，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/)。
- 🔥 **[2024.08.09]** 简化安装方式，支持pypi安装vlmeval相关依赖；优化多模态模型评测体验，基于OpenAI API方式的评测链路，最高加速10倍。
- 🔥 **[2024.07.31]** 重要修改：`llmuses`包名修改为`evalscope`，请同步修改您的代码。
- 🔥 **[2024.07.26]** 支持**VLMEvalKit**作为第三方评测框架，发起多模态模型评测任务。
- 🔥 **[2024.06.29]** 支持**OpenCompass**作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评测任务配置。
- 🔥 **[2024.06.13]** EvalScope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 。
- 🔥 **[2024.06.13]** 接入Agent评测集ToolBench。
</details>

## ❤️ 社区与支持

欢迎加入我们的社区，与其他开发者交流并获取帮助。

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  微信群 | 钉钉群
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">



## 🛠️ 环境准备

我们推荐使用 `conda` 创建虚拟环境，并使用 `pip` 安装。

1.  **创建并激活 Conda 环境** (推荐使用 Python 3.10)
    ```shell
    conda create -n evalscope python=3.10
    conda activate evalscope
    ```

2.  **安装 EvalScope**

    - **方式一：通过 PyPI 安装 (推荐)**
      ```shell
      pip install evalscope
      ```

    - **方式二：通过源码安装 (用于开发)**
      ```shell
      git clone https://github.com/modelscope/evalscope.git
      cd evalscope
      pip install -e .
      ```

3.  **安装额外依赖** (可选)
    根据您的需求，安装相应的功能扩展：
    ```shell
    # 性能测试
    pip install 'evalscope[perf]'

    # 可视化App
    pip install 'evalscope[app]'

    # 其他评测后端
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'

    # 安装所有依赖
    pip install 'evalscope[all]'
    ```
    > 如果您通过源码安装，请将 `evalscope` 替换为 `.`，例如 `pip install '.[perf]'`。

> [!NOTE]
> 本项目曾用名 `llmuses`。如果您需要使用 `v0.4.3` 或更早版本，请运行 `pip install llmuses<=0.4.3` 并使用 `from llmuses import ...` 导入。


## 🚀 快速开始

您可以通过**命令行**或 **Python 代码**两种方式启动评测任务。

### 方式1. 使用命令行

在任意路径下执行 `evalscope eval` 命令即可开始评测。以下命令将在 `gsm8k` 和 `arc` 数据集上评测 `Qwen/Qwen2.5-0.5B-Instruct` 模型，每个数据集只取 5 个样本。

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### 方式2. 使用Python代码

使用 `run_task` 函数和 `TaskConfig` 对象来配置和启动评测任务。

```python
from evalscope import run_task TaskConfig

# 配置评测任务
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct'
    datasets=['gsm8k' 'arc']
    limit=5
)

# 启动评测
run_task(task_cfg)
```

<details><summary><b>💡 提示：</b> `run_task` 还支持字典、YAML 或 JSON 文件作为配置。</summary>

**使用 Python 字典**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct'
    'datasets': ['gsm8k' 'arc']
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**使用 YAML 文件** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### 输出结果
评测完成后，您将在终端看到如下格式的报告：
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 📈 进阶用法

### 自定义评测参数

您可以通过命令行参数精细化控制模型加载、推理和数据集配置。

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master" "precision": "torch.float16" "device_map": "auto"}' \
 --generation-config '{"do_sample":true"temperature":0.6"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0 "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: 模型加载参数，如 `revision` `precision` 等。
- `--generation-config`: 模型生成参数，如 `temperature` `max_tokens` 等。
- `--dataset-args`: 数据集配置参数，如 `few_shot_num` 等。

详情请参考 [📖 全部参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。

### 评测在线模型 API

EvalScope 支持评测通过 API 部署的模型服务（如 vLLM 部署的服务）。只需指定服务地址和 API Key 即可。

1.  **启动模型服务** (以 vLLM 为例)
    ```shell
    export VLLM_USE_MODELSCOPE=True
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen2.5 \
      --port 8801
    ```

2.  **运行评测**
    ```shell
    evalscope eval \
     --model qwen2.5 \
     --eval-type openai_api \
     --api-url http://127.0.0.1:8801/v1 \
     --api-key EMPTY \
     --datasets gsm8k \
     --limit 10
    ```

### ⚔️ 竞技场模式 (Arena)

竞技场模式通过模型间的两两对战（Pairwise Battle）来评估模型性能，并给出胜率和排名，非常适合多模型横向对比。

```text
# 评测结果示例
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
详情请参考 [📖 竞技场模式使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。

### 🖊️ 自定义数据集评测

EvalScope 允许您轻松添加和评测自己的数据集。详情请参考 [📖 自定义数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/index.html)。


## 🧪 其他评测后端
EvalScope 支持通过第三方评测框架（我们称之为“后端”）发起评测任务，以满足多样化的评测需求。

- **Native**: EvalScope 的默认评测框架，功能全面。
- **OpenCompass**: 专注于纯文本评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: 专注于多模态评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: 专注于 RAG 评测，支持 Embedding 和 Reranker 模型。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/index.html)
- **第三方评测工具**: 支持 [ToolBench](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html) 等评测任务。

## ⚡ 推理性能评测工具
EvalScope 提供了一个强大的压力测试工具，用于评估大语言模型服务的性能。

- **关键指标**: 支持吞吐量 (Tokens/s)、首字延迟 (TTFT)、Token 生成延迟 (TPOT) 等。
- **结果记录**: 支持将结果记录到 `wandb` 和 `swanlab`。
- **速度基准**: 可生成类似官方报告的速度基准测试结果。

详情请参考 [📖 性能测试使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)。

输出示例如下：
<p align="center">
    <img src="docs/zh/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>


## 📊 可视化评测结果

EvalScope 提供了一个基于 Gradio 的 WebUI，用于交互式地分析和比较评测结果。

1.  **安装依赖**
    ```bash
    pip install 'evalscope[app]'
    ```

2.  **启动服务**
    ```bash
    evalscope app
    ```
    访问 `http://127.0.0.1:7861` 即可打开可视化界面。

<table>
  <tr>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/setting.png" alt="Setting" style="width: 90%;" />
      <p>设置界面</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/model_compare.png" alt="Model Compare" style="width: 100%;" />
      <p>模型比较</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/report_overview.png" alt="Report Overview" style="width: 100%;" />
      <p>报告概览</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/report_details.png" alt="Report Details" style="width: 91%;" />
      <p>报告详情</p>
    </td>
  </tr>
</table>

详情请参考 [📖 可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)。

## 👷‍♂️ 贡献

我们欢迎来自社区的任何贡献！如果您希望添加新的评测基准、模型或功能，请参考我们的 [贡献指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)。

感谢所有为 EvalScope 做出贡献的开发者！

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>


## 📚 引用

如果您在研究中使用了 EvalScope，请引用我们的工作：
```bibtex
@misc{evalscope_2024
    title={{EvalScope}: Evaluation Framework for Large Models}
    author={ModelScope Team}
    year={2024}
    url={https://github.com/modelscope/evalscope}
}
```


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)

# Arena Mode

Arena mode allows you to configure multiple candidate models and specify a baseline model. The evaluation is conducted through pairwise battles between each candidate model and the baseline model with the win rate and ranking of each model outputted at the end. This approach is suitable for comparative evaluation among multiple models and intuitively reflects the strengths and weaknesses of each model.

## Data Preparation

To support arena mode **all candidate models need to run inference on the same dataset**. The dataset can be a general QA dataset or a domain-specific one. Below is an example using a custom `general_qa` dataset. See the [documentation](../advanced_guides/custom_dataset/llm.md#question-answering-format-qa) for details on using this dataset.

The JSONL file for the `general_qa` dataset should be in the following format. Only the `query` field is required; no additional fields are necessary. Below are two example files:

- Example content of the `arena.jsonl` file:
    ```json
    {"query": "How can I improve my time management skills?"}
    {"query": "What are the most effective ways to deal with stress?"}
    {"query": "What are the main differences between Python and JavaScript programming languages?"}
    {"query": "How can I increase my productivity while working from home?"}
    {"query": "Can you explain the basics of quantum computing?"}
    ```

- Example content of the `example.jsonl` file (with reference answers):
    ```json
    {"query": "What is the capital of France?" "response": "The capital of France is Paris."}
    {"query": "What is the largest mammal in the world?" "response": "The largest mammal in the world is the blue whale."}
    {"query": "How does photosynthesis work?" "response": "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."}
    {"query": "What is the theory of relativity?" "response": "The theory of relativity developed by Albert Einstein describes the laws of physics in relation to observers in different frames of reference."}
    {"query": "Who wrote 'To Kill a Mockingbird'?" "response": "Harper Lee wrote 'To Kill a Mockingbird'."}
    ```

## Candidate Model Inference

After preparing the dataset you can use EvalScope's `run_task` method to perform inference with the candidate models and obtain their outputs for subsequent battles.

Below is an example of how to configure inference tasks for three candidate models: `Qwen2.5-0.5B-Instruct` `Qwen2.5-7B-Instruct` and `Qwen2.5-72B-Instruct` using the same configuration for inference.

Run the following code:
```python
import os
from evalscope import TaskConfig run_task
from evalscope.constants import EvalType

models = ['qwen2.5-72b-instruct' 'qwen2.5-7b-instruct' 'qwen2.5-0.5b-instruct']

task_list = [TaskConfig(
    model=model
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=os.getenv('DASHSCOPE_API_KEY')
    eval_type=EvalType.SERVICE
    datasets=[
        'general_qa'
    ]
    dataset_args={
        'general_qa': {
            'dataset_id': 'custom_eval/text/qa'
            'subset_list': [
                'arena'
                'example'
            ]
        }
    }
    eval_batch_size=10
    generation_config={
        'temperature': 0
        'n': 1
        'max_tokens': 4096
    }) for model in models]

run_task(task_cfg=task_list)
```

<details><summary>Click to view inference results</summary>

Since the `arena` subset does not have reference answers no evaluation metrics are available for this subset. The `example` subset has reference answers so evaluation metrics will be output.
```text
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric          | Subset   |   Num |   Score | Cat.0   |
+=======================+============+=================+==========+=======+=========+=========+
| qwen2.5-0.5b-instruct | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-R       | example  |    12 |  0.8611 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-P       | example  |    12 |  0.1341 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-F       | example  |    12 |  0.1983 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-R       | example  |    12 |  0.55   | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-P       | example  |    12 |  0.0404 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-F       | example  |    12 |  0.0716 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-R       | example  |    12 |  0.8611 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-P       | example  |    12 |  0.1193 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-F       | example  |    12 |  0.1754 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-1          | example  |    12 |  0.1192 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-2          | example  |    12 |  0.0403 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-3          | example  |    12 |  0.0135 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-4          | example  |    12 |  0.0079 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-P       | example  |    12 |  0.1149 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-F       | example  |    12 |  0.1612 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-R       | example  |    12 |  0.6833 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-P       | example  |    12 |  0.0813 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-F       | example  |    12 |  0.1027 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-P       | example  |    12 |  0.101  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-F       | example  |    12 |  0.1361 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-1          | example  |    12 |  0.1009 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-2          | example  |    12 |  0.0807 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-3          | example  |    12 |  0.0625 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-4          | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-P       | example  |    12 |  0.104  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-F       | example  |    12 |  0.1418 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-R       | example  |    12 |  0.7    | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-P       | example  |    12 |  0.078  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-F       | example  |    12 |  0.0964 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-P       | example  |    12 |  0.0942 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-F       | example  |    12 |  0.1235 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-1          | example  |    12 |  0.0939 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-2          | example  |    12 |  0.0777 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-3          | example  |    12 |  0.0625 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-4          | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
```
</details>

## Candidate Model Battles

Next you can use EvalScope's `general_arena` method to conduct battles among candidate models and get their win rates and rankings on each subset. To achieve robust automatic battles you need to configure an LLM as the judge that compares the outputs of models.

During evaluation EvalScope will automatically parse the public evaluation set of candidate models use the judge model to compare the output of each candidate model with the baseline and determine which is better (to avoid model bias outputs are swapped for two rounds per comparison). The judge model's outputs are parsed as win draw or loss and each candidate model's **Elo score** and **win rate** are calculated.

Run the following code:
```python
import os
from evalscope import TaskConfig run_task

task_cfg = TaskConfig(
    model_id='Arena'  # Model ID is 'Arena'; you can omit specifying model ID
    datasets=[
        'general_arena'  # Must be 'general_arena' indicating arena mode
    ]
    dataset_args={
        'general_arena': {
            # 'system_prompt': 'xxx' # Optional: customize the judge model's system prompt here
            # 'prompt_template': 'xxx' # Optional: customize the judge model's prompt template here
            'extra_params':{
                # Configure candidate model names and corresponding report paths
                # Report paths refer to the output paths from the previous step for parsing model inference results
                'models':[
                    {
                        'name': 'qwen2.5-0.5b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-0.5b-instruct'
                    }
                    {
                        'name': 'qwen2.5-7b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-7b-instruct'
                    }
                    {
                        'name': 'qwen2.5-72b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-72b-instruct'
                    }
                ]
                # Set baseline model must be one of the candidate models
                'baseline': 'qwen2.5-7b'
            }
        }
    }
    # Configure judge model parameters
    judge_model_args={
        'model_id': 'qwen-plus'
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        'api_key': os.getenv('DASHSCOPE_API_KEY')
        'generation_config': {
            'temperature': 0.0
            'max_tokens': 8000
        }
    }
    judge_worker_num=5
    # use_cache='outputs/xxx' # Optional: to add new candidate models to existing results specify the existing results path
)

run_task(task_cfg=task_cfg)
```

<details><summary>Click to view evaluation results</summary>

```text
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Model   | Dataset       | Metric        | Subset                                     |   Num |   Score | Cat.0   |
+=========+===============+===============+============================================+=======+=========+=========+
| Arena   | general_arena | winrate       | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0185 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.5469 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.075  | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.8382 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | OVERALL                                    |    44 |  0.3617 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0185 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.3906 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.025  | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.7276 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | OVERALL                                    |    44 |  0.2826 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0909 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.6875 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.0909 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.9412 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | OVERALL                                    |    44 |  0.4469 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+ 
```
</details>


The automatically generated model leaderboard is as follows (output file located in `outputs/xxx/reports/Arena/leaderboard.txt`):

The leaderboard is sorted by win rate in descending order. As shown the `qwen2.5-72b` model performs best across all subsets with the highest win rate while the `qwen2.5-0.5b` model performs the worst.

```text
=== OVERALL LEADERBOARD ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)

=== DATASET LEADERBOARD: general_qa ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)

=== SUBSET LEADERBOARD: general_qa - example ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            54.7  (-15.6 / +14.1)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            1.8  (+0.0 / +7.2)

=== SUBSET LEADERBOARD: general_qa - arena ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            83.8  (-11.1 / +10.3)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            7.5  (-5.0 / +1.6)
```

## Visualization of Battle Results

To intuitively display the results of the battles between candidate models and the baseline EvalScope provides a visualization feature allowing you to compare the results of each candidate model against the baseline model for each sample.

Run the command below to launch the visualization interface:
```shell
evalscope app
```
Open `http://localhost:7860` in your browser to view the visualization page.

Workflow:
1. Select the latest `general_arena` evaluation report and click the "Load and View" button.
2. Click dataset details and select the battle results between your candidate model and the baseline.
3. Adjust the threshold to filter battle results (normalized scores range from 0-1; 0.5 indicates a tie scores above 0.5 indicate the candidate is better than the baseline below 0.5 means worse).

Example below: a battle between `qwen2.5-72b` and `qwen2.5-7b`. The model judged the 72b as better:

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/arena_example.jpg)


# Sandbox Environment Usage

To complete LLM code capability evaluation we need to set up an independent evaluation environment to avoid executing erroneous code in the development environment and causing unavoidable losses. Currently EvalScope has integrated the [ms-enclave](https://github.com/modelscope/ms-enclave) sandbox environment allowing users to evaluate model code capabilities in a controlled environment such as using evaluation benchmarks like HumanEval and LiveCodeBench.

The following introduces two different sandbox usage methods:

- Local usage: Set up the sandbox environment on a local machine and conduct evaluation locally requiring Docker support on the local machine;
- Remote usage: Set up the sandbox environment on a remote server and conduct evaluation through API interfaces requiring Docker support on the remote machine.

## 1. Local Usage

Use Docker to set up a sandbox environment on a local machine and conduct evaluation locally requiring Docker support on the local machine.

### Environment Setup

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in your local Python environment:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration
When running evaluations add the `use_sandbox` and `sandbox_type` parameters to automatically enable the sandbox environment. Other parameters remain the same as regular evaluations:

Here's a complete example code for model evaluation on HumanEval:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig run_task

task_config = TaskConfig(
    model='qwen-plus'
    datasets=['humaneval']
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=env.get('DASHSCOPE_API_KEY')
    eval_type='openai_api'
    eval_batch_size=5
    limit=5
    generation_config={
        'max_tokens': 4096
        'temperature': 0.0
        'seed': 42
    }
    use_sandbox=True # enable sandbox
    sandbox_type='docker' # specify sandbox type
    judge_worker_num=5 # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation EvalScope will automatically start and manage the sandbox environment ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] Local sandbox manager started
...
```

## 2. Remote Usage

Set up the sandbox environment on a remote server and conduct evaluation through API interfaces requiring Docker support on the remote machine.

### Environment Setup

You need to install and configure separately on both the remote machine and local machine.

#### Remote Machine

The environment installation on the remote machine is similar to the local usage method described above:

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in remote Python environment:

```bash
pip install evalscope[sandbox]
```

3. **Start sandbox server**: Run the following command to start the sandbox server:

```bash
ms-enclave server --host 0.0.0.0 --port 1234
```

#### Local Machine

The local machine does not need Docker installation at this point but needs to install EvalScope:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration

When running evaluations add the `use_sandbox` parameter to automatically enable the sandbox environment and specify the remote sandbox server's API address in `sandbox_manager_config`:

Complete example code is as follows:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig run_task

task_config = TaskConfig(
    model='qwen-plus'
    datasets=['humaneval']
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=env.get('DASHSCOPE_API_KEY')
    eval_type='openai_api'
    eval_batch_size=5
    limit=5
    generation_config={
        'max_tokens': 4096
        'temperature': 0.0
        'seed': 42
    }
    use_sandbox=True # enable sandbox
    sandbox_type='docker' # specify sandbox type
    sandbox_manager_config={
        'base_url': 'http://<remote_host>:1234'  # remote sandbox manager URL
    }
    judge_worker_num=5 # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation EvalScope will communicate with the remote sandbox server through API ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] HTTP sandbox manager started connected to http://<remote_host>:1234
...
```


# EvalScope Service Deployment

## Introduction

EvalScope service mode provides HTTP API-based evaluation and stress testing capabilities designed to address the following scenarios:

1. **Remote Invocation**: Support remote evaluation functionality through network without configuring complex evaluation environments locally
2. **Service Integration**: Easily integrate evaluation capabilities into existing workflows CI/CD pipelines or automated testing systems
3. **Multi-user Collaboration**: Support multiple users or systems calling the evaluation service simultaneously improving resource utilization
4. **Unified Management**: Centrally manage evaluation resources and configurations for easier maintenance and monitoring
5. **Flexible Deployment**: Can be deployed on dedicated servers or container environments decoupled from business systems

The Flask service encapsulates EvalScope's core evaluation (eval) and stress testing (perf) functionalities providing services through standard RESTful APIs making evaluation capabilities callable and integrable like other microservices.

## Features

- **Model Evaluation** (`/api/v1/eval`): Support evaluation of OpenAI API-compatible models
- **Performance Testing** (`/api/v1/perf`): Support performance benchmarking of OpenAI API-compatible models
- **Parameter Query**: Provide parameter description endpoints

## Environment Setup


### Full Installation (Recommended)

```bash
pip install evalscope[service]
```

### Development Environment Installation

```bash
# Clone repository
git clone https://github.com/modelscope/evalscope.git
cd evalscope

# Install development version with service
pip install -e '.[service]'
```

## Starting the Service

### Command Line Launch

```bash
# Use default configuration (host: 0.0.0.0 port: 9000)
evalscope service

# Custom host and port
evalscope service --host 127.0.0.1 --port 9000

# Enable debug mode
evalscope service --debug
```

### Python Code Launch

```python
from evalscope.service import run_service

# Start service
run_service(host='0.0.0.0' port=9000 debug=False)
```

## API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response Example:**
```json
{
  "status": "ok"
  "service": "evalscope"
  "timestamp": "2025-12-04T10:00:00"
}
```

### 2. Model Evaluation

```bash
POST /api/v1/eval
```

**Request Body Example:**
```json
{
  "model": "qwen-plus"
  "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
  "api_key": "your-api-key"
  "datasets": ["gsm8k" "iquiz"]
  "limit": 10
  "generation_config": {
    "temperature": 0.0
    "max_tokens": 2048
  }
}
```

**Required Parameters:**
- `model`: Model name
- `datasets`: List of datasets
- `api_url`: API endpoint URL (OpenAI-compatible)

**Optional Parameters:**
- `api_key`: API key (default: "EMPTY")
- `limit`: Evaluation sample quantity limit
- `eval_batch_size`: Batch size (default: 1)
- `generation_config`: Generation configuration
  - `temperature`: Temperature parameter (default: 0.0)
  - `max_tokens`: Maximum generation tokens (default: 2048)
  - `top_p`: Nucleus sampling parameter
  - `top_k`: Top-k sampling parameter
- `work_dir`: Output directory
- `debug`: Debug mode
- `seed`: Random seed (default: 42)

**Response Example:**
```json
{
  "status": "success"
  "message": "Evaluation completed"
  "result": {"...": "..."}
  "output_dir": "/path/to/outputs/20251204_100000"
}
```

### 3. Performance Testing

```bash
POST /api/v1/perf
```

**Request Body Example:**
```json
{
  "model": "qwen-plus"
  "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  "api": "openai"
  "api_key": "your-api-key"
  "number": 100
  "parallel": 10
  "dataset": "openqa"
  "max_tokens": 2048
  "temperature": 0.0
}
```

**Required Parameters:**
- `model`: Model name
- `url`: Complete API endpoint URL

**Optional Parameters:**
- `api`: API type (openai/dashscope/anthropic/gemini default: "openai")
- `api_key`: API key
- `number`: Total number of requests (default: 1000)
- `parallel`: Concurrency level (default: 1)
- `rate`: Requests per second limit (default: -1 unlimited)
- `dataset`: Dataset name (default: "openqa")
- `max_tokens`: Maximum generation tokens (default: 2048)
- `temperature`: Temperature parameter (default: 0.0)
- `stream`: Whether to use streaming output (default: true)
- `debug`: Debug mode

**Response Example:**
```json
{
  "status": "success"
  "message": "Performance test completed"
  "output_dir": "/path/to/outputs"
  "results": {
    "parallel_10_number_100": {
      "metrics": {"...": "..."}
      "percentiles": {"...": "..."}
    }
  }
}
```

### 4. Get Evaluation Parameter Description

```bash
GET /api/v1/eval/params
```

Returns descriptions of all parameters supported by the evaluation endpoint.

### 5. Get Performance Test Parameter Description

```bash
GET /api/v1/perf/params
```

Returns descriptions of all parameters supported by the performance test endpoint.

## Usage Examples

### Testing Evaluation Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "api_key": "your-api-key"
    "datasets": ["gsm8k"]
    "limit": 5
  }'
```

### Testing Performance Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/perf \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    "api": "openai"
    "number": 50
    "parallel": 5
  }'
```

### Using Python requests

```python
import requests

# Evaluation request
eval_response = requests.post(
    'http://localhost:9000/api/v1/eval'
    json={
        'model': 'qwen-plus'
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        'api_key': 'your-api-key'
        'datasets': ['gsm8k' 'iquiz']
        'limit': 10
        'generation_config': {
            'temperature': 0.0
            'max_tokens': 2048
        }
    }
)
print(eval_response.json())

# Performance test request
perf_response = requests.post(
    'http://localhost:9000/api/v1/perf'
    json={
        'model': 'qwen-plus'
        'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        'api': 'openai'
        'number': 100
        'parallel': 10
        'dataset': 'openqa'
    }
)
print(perf_response.json())
```

## Important Notes

1. **OpenAI API-Compatible Models Only**: This service is designed specifically for OpenAI API-compatible models
2. **Long-Running Tasks**: Evaluation and performance testing tasks may take considerable time. We recommend setting appropriate HTTP timeout values on the client side as the API calls are synchronous and will block until completion.
3. **Output Directory**: Evaluation results are saved in the configured `work_dir` default is `outputs/`
4. **Error Handling**: The service returns detailed error messages and stack traces (in debug mode)
5. **Resource Management**: Pay attention to concurrency settings during stress testing to avoid server overload

## Error Codes

- `400`: Invalid request parameters
- `404`: Endpoint not found
- `500`: Internal server error

## Example Scenarios

### Scenario 1: Quick Evaluation of Qwen Model

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "api_key": "sk-..."
    "datasets": ["gsm8k"]
    "limit": 100
  }'
```

### Scenario 2: Stress Testing Locally Deployed Model

```bash
curl -X POST http://localhost:9000/api/v1/perf \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5"
    "url": "http://localhost:8000/v1/chat/completions"
    "api": "openai"
    "number": 1000
    "parallel": 20
    "max_tokens": 2048
  }'
```

### Scenario 3: Multi-Dataset Evaluation

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "datasets": ["gsm8k" "iquiz" "ceval"]
    "limit": 50
    "eval_batch_size": 4
  }'
```

<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  Chinese Documentation</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documentation</a>
<p>


> ⭐ If you like this project please click the "Star" button in the upper right corner to support us. Your support is our motivation to move forward!

## 📝 Introduction

EvalScope is a powerful and easily extensible model evaluation framework created by the [ModelScope Community](https://modelscope.cn/) aiming to provide a one-stop evaluation solution for large model developers.

Whether you want to evaluate the general capabilities of models conduct multi-model performance comparisons or need to stress test models EvalScope can meet your needs.

## ✨ Key Features

- **📚 Comprehensive Evaluation Benchmarks**: Built-in multiple industry-recognized evaluation benchmarks including MMLU C-Eval GSM8K and more.
- **🧩 Multi-modal and Multi-domain Support**: Supports evaluation of various model types including Large Language Models (LLM) Vision Language Models (VLM) Embedding Reranker AIGC and more.
- **🚀 Multi-backend Integration**: Seamlessly integrates multiple evaluation backends including OpenCompass VLMEvalKit RAGEval to meet different evaluation needs.
- **⚡ Inference Performance Testing**: Provides powerful model service stress testing tools supporting multiple performance metrics such as TTFT TPOT.
- **📊 Interactive Reports**: Provides WebUI visualization interface supporting multi-dimensional model comparison report overview and detailed inspection.
- **⚔️ Arena Mode**: Supports multi-model battles (Pairwise Battle) intuitively ranking and evaluating models.
- **🔧 Highly Extensible**: Developers can easily add custom datasets models and evaluation metrics.

<details><summary>🏛️ Overall Architecture</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope Overall Architecture.
</p>

1.  **Input Layer**
    - **Model Sources**: API models (OpenAI API) Local models (ModelScope)
    - **Datasets**: Standard evaluation benchmarks (MMLU/GSM8k etc.) Custom data (MCQ/QA)

2.  **Core Functions**
    - **Multi-backend Evaluation**: Native backend OpenCompass MTEB VLMEvalKit RAGAS
    - **Performance Monitoring**: Supports multiple model service APIs and data formats tracking TTFT/TPOP and other metrics
    - **Tool Extensions**: Integrates Tool-Bench Needle-in-a-Haystack etc.

3.  **Output Layer**
    - **Structured Reports**: Supports JSON Table Logs
    - **Visualization Platform**: Supports Gradio Wandb SwanLab

</details>

## 🎉 What's New

> [!IMPORTANT]
> **Version 1.0 Refactoring**
>
> Version 1.0 introduces a major overhaul of the evaluation framework establishing a new more modular and extensible API layer under `evalscope/api`. Key improvements include standardized data models for benchmarks samples and results; a registry-based design for components such as benchmarks and metrics; and a rewritten core evaluator that orchestrates the new architecture. Existing benchmark adapters have been migrated to this API resulting in cleaner more consistent and easier-to-maintain implementations.

- 🔥 **[2025.12.02]** Added support for custom multimodal VQA evaluation; refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html). Added support for visualizing model service stress testing in ClearML; refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#clearml).
- 🔥 **[2025.11.26]** Added support for OpenAI-MRCR GSM8K-V MGSM MicroVQA IFBench SciCode benchmarks.
- 🔥 **[2025.11.18]** Added support for custom Function-Call (tool invocation) datasets to test whether models can timely and correctly call tools. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#function-calling-format-fc).
- 🔥 **[2025.11.14]** Added support for SWE-bench_Verified SWE-bench_Lite SWE-bench_Verified_mini code evaluation benchmarks. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html).
- 🔥 **[2025.11.12]** Added `pass@k` `vote@k` `pass^k` and other metric aggregation methods; added support for multimodal evaluation benchmarks such as A_OKVQA CMMU ScienceQA V*Bench.
- 🔥 **[2025.11.07]** Added support for τ²-bench an extended and enhanced version of τ-bench that includes a series of code fixes and adds telecom domain troubleshooting scenarios. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/tau2_bench.html).
- 🔥 **[2025.10.30]** Added support for BFCL-v4 enabling evaluation of agent capabilities including web search and long-term memory. See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v4.html).
- 🔥 **[2025.10.27]** Added support for LogiQA HaluEval MathQA MRI-QA PIQA QASC CommonsenseQA and other evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.26]** Added support for Conll-2003 CrossNER Copious GeniaNER HarveyNER MIT-Movie-Trivia MIT-Restaurant OntoNotes5 WNUT2017 and other Named Entity Recognition evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.21]** Optimized sandbox environment usage in code evaluation supporting both local and remote operation modes. For details refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html).
- 🔥 **[2025.10.20]** Added support for evaluation benchmarks including PolyMath SimpleVQA MathVerse MathVision AA-LCR; optimized evalscope perf performance to align with vLLM Bench. For details refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/vs_vllm_bench.html).
- 🔥 **[2025.10.14]** Added support for OCRBench OCRBench-v2 DocVQA InfoVQA ChartQA and BLINK multimodal image-text evaluation benchmarks.
- 🔥 **[2025.09.22]** Code evaluation benchmarks (HumanEval LiveCodeBench) now support running in a sandbox environment. To use this feature please install [ms-enclave](https://github.com/modelscope/ms-enclave) first.
- 🔥 **[2025.09.19]** Added support for multimodal image-text evaluation benchmarks including RealWorldQA AI2D MMStar MMBench and OmniBench as well as pure text evaluation benchmarks such as Multi-IF HealthBench and AMC.
- 🔥 **[2025.09.05]** Added support for vision-language multimodal model evaluation tasks such as MathVista and MMMU. For more supported datasets please [refer to the documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/vlm.html).
- 🔥 **[2025.09.04]** Added support for image editing task evaluation including the [GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) benchmark. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/image_edit.html).
- 🔥 **[2025.08.22]** Version 1.0 Refactoring. Break changes please [refer to](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#switching-to-version-v1-0).
<details><summary>More</summary>

- 🔥 **[2025.07.18]** The model stress testing now supports randomly generating image-text data for multimodal model evaluation. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#id4).
- 🔥 **[2025.07.16]** Support for [τ-bench](https://github.com/sierra-research/tau-bench) has been added enabling the evaluation of AI Agent performance and reliability in real-world scenarios involving dynamic user and tool interactions. For usage instructions please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html#bench).
- 🔥 **[2025.07.14]** Support for "Humanity's Last Exam" ([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle)) a highly challenging evaluation benchmark. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam).
- 🔥 **[2025.07.03]** Refactored Arena Mode: now supports custom model battles outputs a model leaderboard and provides battle result visualization. See [reference](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html) for details.
- 🔥 **[2025.06.28]** Optimized custom dataset evaluation: now supports evaluation without reference answers. Enhanced LLM judge usage with built-in modes for "scoring directly without reference answers" and "checking answer consistency with reference answers". See [reference](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#qa) for details.
- 🔥 **[2025.06.19]** Added support for the [BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3) benchmark designed to evaluate model function-calling capabilities across various scenarios. For more information refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html).
- 🔥 **[2025.06.02]** Added support for the Needle-in-a-Haystack test. Simply specify `needle_haystack` to conduct the test and a corresponding heatmap will be generated in the `outputs/reports` folder providing a visual representation of the model's performance. Refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/needle_haystack.html) for more details.
- 🔥 **[2025.05.29]** Added support for two long document evaluation benchmarks: [DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary) and [FRAMES](https://modelscope.cn/datasets/iic/frames/summary). For usage guidelines please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html).
- 🔥 **[2025.05.16]** Model service performance stress testing now supports setting various levels of concurrency and outputs a performance test report. [Reference example](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html#id3).
- 🔥 **[2025.05.13]** Added support for the [ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static) dataset to evaluate model's tool-calling capabilities. Refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html) for usage instructions. Also added support for the [DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview) and [Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val) benchmarks to assess the reasoning capabilities of models.
- 🔥 **[2025.04.29]** Added Qwen3 Evaluation Best Practices [welcome to read 📖](https://evalscope.readthedocs.io/en/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** Support for text-to-image evaluation: Supports 8 metrics including MPS HPSv2.1Score etc. and evaluation benchmarks such as EvalMuse GenAI-Bench. Refer to the [user documentation](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/t2i.html) for more details.
- 🔥 **[2025.04.10]** Model service stress testing tool now supports the `/v1/completions` endpoint (the default endpoint for vLLM benchmarking)
- 🔥 **[2025.04.08]** Support for evaluating embedding model services compatible with the OpenAI API has been added. For more details check the [user guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters).
- 🔥 **[2025.03.27]** Added support for [AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview) and [ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary) evaluation benchmarks. For usage notes please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** The model inference service stress testing now supports generating prompts of specified length using random values. Refer to the [user guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#using-the-random-dataset) for more details.
- 🔥 **[2025.03.13]** Added support for the [LiveCodeBench](https://www.modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary) code evaluation benchmark which can be used by specifying `live_code_bench`. Supports evaluating QwQ-32B on LiveCodeBench refer to the [best practices](https://evalscope.readthedocs.io/en/latest/best_practice/eval_qwq.html).
- 🔥 **[2025.03.11]** Added support for the [SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary) and [Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary) evaluation benchmarks. These are used to assess the factual accuracy of models and you can specify `simple_qa` and `chinese_simpleqa` for use. Support for specifying a judge model is also available. For more details refer to the [relevant parameter documentation](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).
- 🔥 **[2025.03.07]** Added support for the [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/summary) model evaluate the model's reasoning ability and reasoning efficiency refer to [📖 Best Practices for QwQ-32B Evaluation](https://evalscope.readthedocs.io/en/latest/best_practice/eval_qwq.html) for more details.
- 🔥 **[2025.03.04]** Added support for the [SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary) dataset which covers 13 categories 72 first-level disciplines and 285 second-level disciplines totaling 26529 questions. You can use it by specifying `super_gpqa`.
- 🔥 **[2025.03.03]** Added support for evaluating the IQ and EQ of models. Refer to [📖 Best Practices for IQ and EQ Evaluation](https://evalscope.readthedocs.io/en/latest/best_practice/iquiz.html) to find out how smart your AI is!
- 🔥 **[2025.02.27]** Added support for evaluating the reasoning efficiency of models. Refer to [📖 Best Practices for Evaluating Thinking Efficiency](https://evalscope.readthedocs.io/en/latest/best_practice/think_eval.html). This implementation is inspired by the works [Overthinking](https://doi.org/10.48550/arXiv.2412.21187) and [Underthinking](https://doi.org/10.48550/arXiv.2501.18585).
- 🔥 **[2025.02.25]** Added support for two model inference-related evaluation benchmarks: [MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR) and [ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary). To use them simply specify `musr` and `process_bench` respectively in the datasets parameter.
- 🔥 **[2025.02.18]** Supports the AIME25 dataset which contains 15 questions (Grok3 scored 93 on this dataset).
- 🔥 **[2025.02.13]** Added support for evaluating DeepSeek distilled models including AIME24 MATH-500 and GPQA-Diamond datasets，refer to [best practice](https://evalscope.readthedocs.io/en/latest/best_practice/deepseek_r1_distill.html); Added support for specifying the `eval_batch_size` parameter to accelerate model evaluation.
- 🔥 **[2025.01.20]** Support for visualizing evaluation results including single model evaluation results and multi-model comparison refer to the [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html) for more details; Added [`iquiz`](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary) evaluation example evaluating the IQ and EQ of the model.
- 🔥 **[2025.01.07]** Native backend: Support for model API evaluation is now available. Refer to the [📖 Model API Evaluation Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#api) for more details. Additionally support for the `ifeval` evaluation benchmark has been added.
- 🔥🔥 **[2024.12.31]** Support for adding benchmark evaluations refer to the [📖 Benchmark Evaluation Addition Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html); support for custom mixed dataset evaluations allowing for more comprehensive model evaluations with less data refer to the [📖 Mixed Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/collection/index.html).
- 🔥 **[2024.12.13]** Model evaluation optimization: no need to pass the `--template-type` parameter anymore; supports starting evaluation with `evalscope eval --args`. Refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html) for more details.
- 🔥 **[2024.11.26]** The model inference service performance evaluator has been completely refactored: it now supports local inference service startup and Speed Benchmark; asynchronous call error handling has been optimized. For more details refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).
- 🔥 **[2024.10.31]** The best practice for evaluating Multimodal-RAG has been updated please check the [📖 Blog](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag) for more details.
- 🔥 **[2024.10.23]** Supports multimodal RAG evaluation including the assessment of image-text retrieval using [CLIP_Benchmark](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/clip_benchmark.html) and extends [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html) to support end-to-end multimodal metrics evaluation.
- 🔥 **[2024.10.8]** Support for RAG evaluation including independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html) as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html).
- 🔥 **[2024.09.18]** Our documentation has been updated to include a blog module featuring some technical research and discussions related to evaluations. We invite you to [📖 read it](https://evalscope.readthedocs.io/en/refact_readme/blog/index.html).
- 🔥 **[2024.09.12]** Support for LongWriter evaluation which supports 10000+ word generation. You can use the benchmark [LongBench-Write](evalscope/third_party/longbench_write/README.md) to measure the long output quality as well as the output length.
- 🔥 **[2024.08.30]** Support for custom dataset evaluations including text datasets and multimodal image-text datasets.
- 🔥 **[2024.08.20]** Updated the official documentation including getting started guides best practices and FAQs. Feel free to [📖read it here](https://evalscope.readthedocs.io/en/latest/)!
- 🔥 **[2024.08.09]** Simplified the installation process allowing for pypi installation of vlmeval dependencies; optimized the multimodal model evaluation experience achieving up to 10x acceleration based on the OpenAI API evaluation chain.
- 🔥 **[2024.07.31]** Important change: The package name `llmuses` has been changed to `evalscope`. Please update your code accordingly.
- 🔥 **[2024.07.26]** Support for **VLMEvalKit** as a third-party evaluation framework to initiate multimodal model evaluation tasks.
- 🔥 **[2024.06.29]** Support for **OpenCompass** as a third-party evaluation framework which we have encapsulated at a higher level supporting pip installation and simplifying evaluation task configuration.
- 🔥 **[2024.06.13]** EvalScope seamlessly integrates with the fine-tuning framework SWIFT providing full-chain support from LLM training to evaluation.
- 🔥 **[2024.06.13]** Integrated the Agent evaluation dataset ToolBench.

</details>

## ❤️ Community & Support

Welcome to join our community to communicate with other developers and get help.

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  WeChat Group | DingTalk Group
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">



## 🛠️ Environment Setup

We recommend using `conda` to create a virtual environment and install with `pip`.

1.  **Create and Activate Conda Environment** (Python 3.10 recommended)
    ```shell
    conda create -n evalscope python=3.10
    conda activate evalscope
    ```

2.  **Install EvalScope**

    - **Method 1: Install via PyPI (Recommended)**
      ```shell
      pip install evalscope
      ```

    - **Method 2: Install from Source (For Development)**
      ```shell
      git clone https://github.com/modelscope/evalscope.git
      cd evalscope
      pip install -e .
      ```

3.  **Install Additional Dependencies** (Optional)
    Install corresponding feature extensions according to your needs:
    ```shell
    # Performance testing
    pip install 'evalscope[perf]'

    # Visualization App
    pip install 'evalscope[app]'

    # Other evaluation backends
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'

    # Install all dependencies
    pip install 'evalscope[all]'
    ```
    > If you installed from source please replace `evalscope` with `.` for example `pip install '.[perf]'`.

> [!NOTE]
> This project was formerly known as `llmuses`. If you need to use `v0.4.3` or earlier versions please run `pip install llmuses<=0.4.3` and use `from llmuses import ...` for imports.


## 🚀 Quick Start

You can start evaluation tasks in two ways: **command line** or **Python code**.

### Method 1. Using Command Line

Execute the `evalscope eval` command in any path to start evaluation. The following command will evaluate the `Qwen/Qwen2.5-0.5B-Instruct` model on `gsm8k` and `arc` datasets taking only 5 samples from each dataset.

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### Method 2. Using Python Code

Use the `run_task` function and `TaskConfig` object to configure and start evaluation tasks.

```python
from evalscope import run_task TaskConfig

# Configure evaluation task
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct'
    datasets=['gsm8k' 'arc']
    limit=5
)

# Start evaluation
run_task(task_cfg)
```

<details><summary><b>💡 Tip:</b> `run_task` also supports dictionaries YAML or JSON files as configuration.</summary>

**Using Python Dictionary**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct'
    'datasets': ['gsm8k' 'arc']
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**Using YAML File** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### Output Results
After evaluation completion you will see a report in the terminal in the following format:
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 📈 Advanced Usage

### Custom Evaluation Parameters

You can fine-tune model loading inference and dataset configuration through command line parameters.

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master" "precision": "torch.float16" "device_map": "auto"}' \
 --generation-config '{"do_sample":true"temperature":0.6"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0 "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: Model loading parameters such as `revision` `precision` etc.
- `--generation-config`: Model generation parameters such as `temperature` `max_tokens` etc.
- `--dataset-args`: Dataset configuration parameters such as `few_shot_num` etc.

For details please refer to [📖 Complete Parameter Guide](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).

### Evaluating Online Model APIs

EvalScope supports evaluating model services deployed via APIs (such as services deployed with vLLM). Simply specify the service address and API Key.

1.  **Start Model Service** (using vLLM as example)
    ```shell
    export VLLM_USE_MODELSCOPE=True
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen2.5 \
      --port 8801
    ```

2.  **Run Evaluation**
    ```shell
    evalscope eval \
     --model qwen2.5 \
     --eval-type openai_api \
     --api-url http://127.0.0.1:8801/v1 \
     --api-key EMPTY \
     --datasets gsm8k \
     --limit 10
    ```

### ⚔️ Arena Mode

Arena mode evaluates model performance through pairwise battles between models providing win rates and rankings perfect for horizontal comparison of multiple models.

```text
# Example evaluation results
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
For details please refer to [📖 Arena Mode Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html).

### 🖊️ Custom Dataset Evaluation

EvalScope allows you to easily add and evaluate your own datasets. For details please refer to [📖 Custom Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html).


## 🧪 Other Evaluation Backends
EvalScope supports launching evaluation tasks through third-party evaluation frameworks (we call them "backends") to meet diverse evaluation needs.

- **Native**: EvalScope's default evaluation framework with comprehensive functionality.
- **OpenCompass**: Focuses on text-only evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: Focuses on multi-modal evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: Focuses on RAG evaluation supporting Embedding and Reranker models. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/index.html)
- **Third-party Evaluation Tools**: Supports evaluation tasks like [ToolBench](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html).

## ⚡ Inference Performance Evaluation Tool
EvalScope provides a powerful stress testing tool for evaluating the performance of large language model services.

- **Key Metrics**: Supports throughput (Tokens/s) first token latency (TTFT) token generation latency (TPOT) etc.
- **Result Recording**: Supports recording results to `wandb` and `swanlab`.
- **Speed Benchmarks**: Can generate speed benchmark results similar to official reports.

For details please refer to [📖 Performance Testing Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).

Example output is shown below:
<p align="center">
    <img src="docs/en/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>


## 📊 Visualizing Evaluation Results

EvalScope provides a Gradio-based WebUI for interactive analysis and comparison of evaluation results.

1.  **Install Dependencies**
    ```bash
    pip install 'evalscope[app]'
    ```

2.  **Start Service**
    ```bash
    evalscope app
    ```
    Visit `http://127.0.0.1:7861` to open the visualization interface.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/setting.png" alt="Setting" style="width: 85%;" />
      <p>Settings Interface</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/model_compare.png" alt="Model Compare" style="width: 100%;" />
      <p>Model Comparison</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/report_overview.png" alt="Report Overview" style="width: 100%;" />
      <p>Report Overview</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/report_details.png" alt="Report Details" style="width: 85%;" />
      <p>Report Details</p>
    </td>
  </tr>
</table>

For details please refer to [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html).

## 👷‍♂️ Contributing

We welcome any contributions from the community! If you want to add new evaluation benchmarks models or features please refer to our [Contributing Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html).

Thanks to all developers who have contributed to EvalScope!

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>


## 📚 Citation

If you use EvalScope in your research please cite our work:
```bibtex
@misc{evalscope_2024
    title={{EvalScope}: Evaluation Framework for Large Models}
    author={ModelScope Team}
    year={2024}
    url={https://github.com/modelscope/evalscope}
}
```


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)

<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  中文 &nbsp ｜ &nbsp <a href="evalscope.md">English</a> &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/zh-cn/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documents</a>
<p>


> ⭐ 如果你喜欢这个项目，请点击右上角的 "Star" 按钮支持我们。你的支持是我们前进的动力！

## 📝 简介

EvalScope 是由[魔搭社区](https://modelscope.cn/)打造的一款功能强大、易于扩展的模型评测框架，旨在为大模型开发者提供一站式评测解决方案。

无论您是想评估模型的通用能力、进行多模型性能对比，还是需要对模型进行压力测试，EvalScope 都能满足您的需求。

## ✨ 主要特性

- **📚 全面的评测基准**: 内置 MMLU C-Eval GSM8K 等多个业界公认的评测基准。
- **🧩 多模态与多领域支持**: 支持大语言模型 (LLM)、多模态 (VLM)、Embedding、Reranker、AIGC 等多种模型的评测。
- **🚀 多后端集成**: 无缝集成 OpenCompass VLMEvalKit RAGEval 等多种评测后端，满足不同评测需求。
- **⚡ 推理性能测试**: 提供强大的模型服务压力测试工具，支持 TTFT TPOT 等多项性能指标。
- **📊 交互式报告**: 提供 WebUI 可视化界面，支持多维度模型对比、报告概览和详情查阅。
- **⚔️ 竞技场模式**: 支持多模型对战 (Pairwise Battle)，直观地对模型进行排名和评估。
- **🔧 高度可扩展**: 开发者可以轻松添加自定义数据集、模型和评测指标。

<details><summary>🏛️ 整体架构</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope 整体架构图.
</p>

1.  **输入层**
    - **模型来源**: API模型（OpenAI API）、本地模型（ModelScope）
    - **数据集**: 标准评测基准（MMLU/GSM8k等）、自定义数据（MCQ/QA）

2.  **核心功能**
    - **多后端评估**: 原生后端、OpenCompass、MTEB、VLMEvalKit、RAGAS
    - **性能监控**: 支持多种模型服务 API 和数据格式，追踪 TTFT/TPOP 等指标
    - **工具扩展**: 集成 Tool-Bench Needle-in-a-Haystack 等

3.  **输出层**
    - **结构化报告**: 支持 JSON Table Logs
    - **可视化平台**: 支持 Gradio Wandb SwanLab

</details>

## 🎉 内容更新

> [!IMPORTANT]
> **版本 1.0 重构**
>
> 版本 1.0 对评测框架进行了重大重构，在 `evalscope/api` 下建立了全新的、更模块化且易扩展的 API 层。主要改进包括：为基准、样本和结果引入了标准化数据模型；对基准和指标等组件采用注册表式设计；并重写了核心评测器以协同新架构。现有的基准已迁移到这一 API，实现更加简洁、一致且易于维护。

- 🔥 **[2025.12.02]** 支持自定义多模态VQA评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html) ；支持模型服务压测在 ClearML 上可视化，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#clearml)。
- 🔥 **[2025.11.26]** 新增支持 OpenAI-MRCR、GSM8K-V、MGSM、MicroVQA、IFBench、SciCode 评测基准。
- 🔥 **[2025.11.18]** 支持自定义 Function-Call（工具调用）数据集，来测试模型能否适时并正确调用工具，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#fc)
- 🔥 **[2025.11.14]** 新增支持SWE-bench_Verified SWE-bench_Lite SWE-bench_Verified_mini 代码评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)。
- 🔥 **[2025.11.12]** 新增`pass@k`、`vote@k`、`pass^k`等指标聚合方法；新增支持A_OKVQA CMMU ScienceQ V*Bench等多模态评测基准。
- 🔥 **[2025.11.07]** 新增支持τ²-bench，是 τ-bench 的扩展与增强版本，包含一系列代码修复，并新增了电信（telecom）领域的故障排查场景，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau2_bench.html)。
- 🔥 **[2025.10.30]** 新增支持BFCL-v4，支持agent的网络搜索和长期记忆能力的评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v4.html)。
- 🔥 **[2025.10.27]** 新增支持LogiQA HaluEval MathQA MRI-QA PIQA QASC CommonsenseQA等评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.26]** 新增支持Conll-2003 CrossNER Copious GeniaNER HarveyNER MIT-Movie-Trivia MIT-Restaurant OntoNotes5 WNUT2017 等命名实体识别评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.21]** 优化代码评测中的沙箱环境使用，支持在本地和远程两种模式下运行，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 🔥 **[2025.10.20]** 新增支持PolyMath SimpleVQA MathVerse MathVision AA-LCR 等评测基准；优化evalscope perf表现，对齐vLLM Bench，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/vs_vllm_bench.html)。
- 🔥 **[2025.10.14]** 新增支持OCRBench OCRBench-v2 DocVQA InfoVQA ChartQA BLINK 等图文多模态评测基准。
- 🔥 **[2025.09.22]** 代码评测基准(HumanEval LiveCodeBench)支持在沙箱环境中运行，要使用该功能需先安装[ms-enclave](https://github.com/modelscope/ms-enclave)。
- 🔥 **[2025.09.19]** 新增支持RealWorldQA、AI2D、MMStar、MMBench、OmniBench等图文多模态评测基准，和Multi-IF、HealthBench、AMC等纯文本评测基准。
- 🔥 **[2025.09.05]** 支持视觉-语言多模态大模型的评测任务，例如：MathVista、MMMU，更多支持数据集请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/vlm.html)。
- 🔥 **[2025.09.04]** 支持图像编辑任务评测，支持[GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) 评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/image_edit.html)。
- 🔥 **[2025.08.22]** Version 1.0 重构，不兼容的更新请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#v1-0)。
<details> <summary>更多</summary>

- 🔥 **[2025.07.18]** 模型压测支持随机生成图文数据，用于多模态模型压测，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#id4)。
- 🔥 **[2025.07.16]** 支持[τ-bench](https://github.com/sierra-research/tau-bench)，用于评估 AI Agent在动态用户和工具交互的实际环境中的性能和可靠性，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#bench)。
- 🔥 **[2025.07.14]** 支持“人类最后的考试”([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle))，这一高难度评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam)。
- 🔥 **[2025.07.03]** 重构了竞技场模式，支持自定义模型对战，输出模型排行榜，以及对战结果可视化，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。
- 🔥 **[2025.06.28]** 优化自定义数据集评测，支持无参考答案评测；优化LLM裁判使用，预置“无参考答案直接打分” 和 “判断答案是否与参考答案一致”两种模式，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa)
- 🔥 **[2025.06.19]** 新增支持[BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3)评测基准，用于评测模型在多种场景下的函数调用能力，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)。
- 🔥 **[2025.06.02]** 新增支持大海捞针测试（Needle-in-a-Haystack），指定`needle_haystack`即可进行测试，并在`outputs/reports`文件夹下生成对应的heatmap，直观展现模型性能，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)。
- 🔥 **[2025.05.29]** 新增支持[DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)和[FRAMES](https://modelscope.cn/datasets/iic/frames/summary)两个长文档评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.05.16]** 模型服务性能压测支持设置多种并发，并输出性能压测报告，[参考示例](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id3)。
- 🔥 **[2025.05.13]** 新增支持[ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static)数据集，评测模型的工具调用能力，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)；支持[DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview)和[Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val)评测基准，评测模型的推理能力。
- 🔥 **[2025.04.29]** 新增Qwen3评测最佳实践，[欢迎阅读📖](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** 支持文生图评测：支持MPS、HPSv2.1Score等8个指标，支持EvalMuse、GenAI-Bench等评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/t2i.html)
- 🔥 **[2025.04.10]** 模型服务压测工具支持`/v1/completions`端点（也是vLLM基准测试的默认端点）
- 🔥 **[2025.04.08]** 支持OpenAI API兼容的Embedding模型服务评测，查看[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters)
- 🔥 **[2025.03.27]** 新增支持[AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview)和[ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** 模型推理服务压测支持random生成指定范围长度的prompt，参考[使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random)
- 🔥 **[2025.03.13]** 新增支持[LiveCodeBench](https://www.modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary)代码评测基准，指定`live_code_bench`即可使用；支持QwQ-32B 在LiveCodeBench上评测，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.11]** 新增支持[SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)和[Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)评测基准，用与评测模型的事实正确性，指定`simple_qa`和`chinese_simpleqa`使用。同时支持指定裁判模型，参考[相关参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。
- 🔥 **[2025.03.07]** 新增QwQ-32B模型评测最佳实践，评测了模型的推理能力以及推理效率，参考[📖QwQ-32B模型评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.04]** 新增支持[SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)数据集，其覆盖 13 个门类、72 个一级学科和 285 个二级学科，共 26529 个问题，指定`super_gpqa`即可使用。
- 🔥 **[2025.03.03]** 新增支持评测模型的智商和情商，参考[📖智商和情商评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/iquiz.html)，来测测你家的AI有多聪明？
- 🔥 **[2025.02.27]** 新增支持评测推理模型的思考效率，参考[📖思考效率评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/think_eval.html)，该实现参考了[Overthinking](https://doi.org/10.48550/arXiv.2412.21187) 和 [Underthinking](https://doi.org/10.48550/arXiv.2501.18585)两篇工作。
- 🔥 **[2025.02.25]** 新增支持[MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR)和[ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary)两个模型推理相关评测基准，datasets分别指定`musr`和`process_bench`即可使用。
- 🔥 **[2025.02.18]** 支持AIME25数据集，包含15道题目（Grok3 在该数据集上得分为93分）
- 🔥 **[2025.02.13]** 支持DeepSeek蒸馏模型评测，包括AIME24 MATH-500 GPQA-Diamond数据集，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/deepseek_r1_distill.html)；支持指定`eval_batch_size`参数，加速模型评测
- 🔥 **[2025.01.20]** 支持可视化评测结果，包括单模型评测结果和多模型评测结果对比，参考[📖可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)；新增[`iquiz`](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)评测样例，评测模型的IQ和EQ。
- 🔥 **[2025.01.07]** Native backend: 支持模型API评测，参考[📖模型API评测指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#api)；新增支持`ifeval`评测基准。
- 🔥🔥 **[2024.12.31]** 支持基准评测添加，参考[📖基准评测添加指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)；支持自定义混合数据集评测，用更少的数据，更全面的评测模型，参考[📖混合数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html)
- 🔥 **[2024.12.13]** 模型评测优化，不再需要传递`--template-type`参数；支持`evalscope eval --args`启动评测，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html)
- 🔥 **[2024.11.26]** 模型推理压测工具重构完成：支持本地启动推理服务、支持Speed Benchmark；优化异步调用错误处理，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)
- 🔥 **[2024.10.31]** 多模态RAG评测最佳实践发布，参考[📖博客](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag)
- 🔥 **[2024.10.23]** 支持多模态RAG评测，包括[CLIP_Benchmark](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/clip_benchmark.html)评测图文检索器，以及扩展了[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)以支持端到端多模态指标评测。
- 🔥 **[2024.10.8]** 支持RAG评测，包括使用[MTEB/CMTEB](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html)进行embedding模型和reranker的独立评测，以及使用[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)进行端到端评测。
- 🔥 **[2024.09.18]** 我们的文档增加了博客模块，包含一些评测相关的技术调研和分享，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/blog/index.html)
- 🔥 **[2024.09.12]** 支持 LongWriter 评测，您可以使用基准测试 [LongBench-Write](evalscope/third_party/longbench_write/README.md) 来评测长输出的质量以及输出长度。
- 🔥 **[2024.08.30]** 支持自定义数据集评测，包括文本数据集和多模态图文数据集。
- 🔥 **[2024.08.20]** 更新了官方文档，包括快速上手、最佳实践和常见问题等，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/)。
- 🔥 **[2024.08.09]** 简化安装方式，支持pypi安装vlmeval相关依赖；优化多模态模型评测体验，基于OpenAI API方式的评测链路，最高加速10倍。
- 🔥 **[2024.07.31]** 重要修改：`llmuses`包名修改为`evalscope`，请同步修改您的代码。
- 🔥 **[2024.07.26]** 支持**VLMEvalKit**作为第三方评测框架，发起多模态模型评测任务。
- 🔥 **[2024.06.29]** 支持**OpenCompass**作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评测任务配置。
- 🔥 **[2024.06.13]** EvalScope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 。
- 🔥 **[2024.06.13]** 接入Agent评测集ToolBench。
</details>

## ❤️ 社区与支持

欢迎加入我们的社区，与其他开发者交流并获取帮助。

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  微信群 | 钉钉群
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">



## 🛠️ 环境准备

我们推荐使用 `conda` 创建虚拟环境，并使用 `pip` 安装。

1.  **创建并激活 Conda 环境** (推荐使用 Python 3.10)
    ```shell
    conda create -n evalscope python=3.10
    conda activate evalscope
    ```

2.  **安装 EvalScope**

    - **方式一：通过 PyPI 安装 (推荐)**
      ```shell
      pip install evalscope
      ```

    - **方式二：通过源码安装 (用于开发)**
      ```shell
      git clone https://github.com/modelscope/evalscope.git
      cd evalscope
      pip install -e .
      ```

3.  **安装额外依赖** (可选)
    根据您的需求，安装相应的功能扩展：
    ```shell
    # 性能测试
    pip install 'evalscope[perf]'

    # 可视化App
    pip install 'evalscope[app]'

    # 其他评测后端
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'

    # 安装所有依赖
    pip install 'evalscope[all]'
    ```
    > 如果您通过源码安装，请将 `evalscope` 替换为 `.`，例如 `pip install '.[perf]'`。

> [!NOTE]
> 本项目曾用名 `llmuses`。如果您需要使用 `v0.4.3` 或更早版本，请运行 `pip install llmuses<=0.4.3` 并使用 `from llmuses import ...` 导入。


## 🚀 快速开始

您可以通过**命令行**或 **Python 代码**两种方式启动评测任务。

### 方式1. 使用命令行

在任意路径下执行 `evalscope eval` 命令即可开始评测。以下命令将在 `gsm8k` 和 `arc` 数据集上评测 `Qwen/Qwen2.5-0.5B-Instruct` 模型，每个数据集只取 5 个样本。

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### 方式2. 使用Python代码

使用 `run_task` 函数和 `TaskConfig` 对象来配置和启动评测任务。

```python
from evalscope import run_task TaskConfig

# 配置评测任务
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct'
    datasets=['gsm8k' 'arc']
    limit=5
)

# 启动评测
run_task(task_cfg)
```

<details><summary><b>💡 提示：</b> `run_task` 还支持字典、YAML 或 JSON 文件作为配置。</summary>

**使用 Python 字典**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct'
    'datasets': ['gsm8k' 'arc']
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**使用 YAML 文件** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### 输出结果
评测完成后，您将在终端看到如下格式的报告：
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 📈 进阶用法

### 自定义评测参数

您可以通过命令行参数精细化控制模型加载、推理和数据集配置。

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master" "precision": "torch.float16" "device_map": "auto"}' \
 --generation-config '{"do_sample":true"temperature":0.6"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0 "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: 模型加载参数，如 `revision` `precision` 等。
- `--generation-config`: 模型生成参数，如 `temperature` `max_tokens` 等。
- `--dataset-args`: 数据集配置参数，如 `few_shot_num` 等。

详情请参考 [📖 全部参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。

### 评测在线模型 API

EvalScope 支持评测通过 API 部署的模型服务（如 vLLM 部署的服务）。只需指定服务地址和 API Key 即可。

1.  **启动模型服务** (以 vLLM 为例)
    ```shell
    export VLLM_USE_MODELSCOPE=True
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen2.5 \
      --port 8801
    ```

2.  **运行评测**
    ```shell
    evalscope eval \
     --model qwen2.5 \
     --eval-type openai_api \
     --api-url http://127.0.0.1:8801/v1 \
     --api-key EMPTY \
     --datasets gsm8k \
     --limit 10
    ```

### ⚔️ 竞技场模式 (Arena)

竞技场模式通过模型间的两两对战（Pairwise Battle）来评估模型性能，并给出胜率和排名，非常适合多模型横向对比。

```text
# 评测结果示例
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
详情请参考 [📖 竞技场模式使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。

### 🖊️ 自定义数据集评测

EvalScope 允许您轻松添加和评测自己的数据集。详情请参考 [📖 自定义数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/index.html)。


## 🧪 其他评测后端
EvalScope 支持通过第三方评测框架（我们称之为“后端”）发起评测任务，以满足多样化的评测需求。

- **Native**: EvalScope 的默认评测框架，功能全面。
- **OpenCompass**: 专注于纯文本评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: 专注于多模态评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: 专注于 RAG 评测，支持 Embedding 和 Reranker 模型。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/index.html)
- **第三方评测工具**: 支持 [ToolBench](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html) 等评测任务。

## ⚡ 推理性能评测工具
EvalScope 提供了一个强大的压力测试工具，用于评估大语言模型服务的性能。

- **关键指标**: 支持吞吐量 (Tokens/s)、首字延迟 (TTFT)、Token 生成延迟 (TPOT) 等。
- **结果记录**: 支持将结果记录到 `wandb` 和 `swanlab`。
- **速度基准**: 可生成类似官方报告的速度基准测试结果。

详情请参考 [📖 性能测试使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)。

输出示例如下：
<p align="center">
    <img src="docs/zh/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>


## 📊 可视化评测结果

EvalScope 提供了一个基于 Gradio 的 WebUI，用于交互式地分析和比较评测结果。

1.  **安装依赖**
    ```bash
    pip install 'evalscope[app]'
    ```

2.  **启动服务**
    ```bash
    evalscope app
    ```
    访问 `http://127.0.0.1:7861` 即可打开可视化界面。

<table>
  <tr>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/setting.png" alt="Setting" style="width: 90%;" />
      <p>设置界面</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/model_compare.png" alt="Model Compare" style="width: 100%;" />
      <p>模型比较</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/report_overview.png" alt="Report Overview" style="width: 100%;" />
      <p>报告概览</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/report_details.png" alt="Report Details" style="width: 91%;" />
      <p>报告详情</p>
    </td>
  </tr>
</table>

详情请参考 [📖 可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)。

## 👷‍♂️ 贡献

我们欢迎来自社区的任何贡献！如果您希望添加新的评测基准、模型或功能，请参考我们的 [贡献指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)。

感谢所有为 EvalScope 做出贡献的开发者！

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>


## 📚 引用

如果您在研究中使用了 EvalScope，请引用我们的工作：
```bibtex
@misc{evalscope_2024
    title={{EvalScope}: Evaluation Framework for Large Models}
    author={ModelScope Team}
    year={2024}
    url={https://github.com/modelscope/evalscope}
}
```


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)

# Arena Mode

Arena mode allows you to configure multiple candidate models and specify a baseline model. The evaluation is conducted through pairwise battles between each candidate model and the baseline model with the win rate and ranking of each model outputted at the end. This approach is suitable for comparative evaluation among multiple models and intuitively reflects the strengths and weaknesses of each model.

## Data Preparation

To support arena mode **all candidate models need to run inference on the same dataset**. The dataset can be a general QA dataset or a domain-specific one. Below is an example using a custom `general_qa` dataset. See the [documentation](../advanced_guides/custom_dataset/llm.md#question-answering-format-qa) for details on using this dataset.

The JSONL file for the `general_qa` dataset should be in the following format. Only the `query` field is required; no additional fields are necessary. Below are two example files:

- Example content of the `arena.jsonl` file:
    ```json
    {"query": "How can I improve my time management skills?"}
    {"query": "What are the most effective ways to deal with stress?"}
    {"query": "What are the main differences between Python and JavaScript programming languages?"}
    {"query": "How can I increase my productivity while working from home?"}
    {"query": "Can you explain the basics of quantum computing?"}
    ```

- Example content of the `example.jsonl` file (with reference answers):
    ```json
    {"query": "What is the capital of France?" "response": "The capital of France is Paris."}
    {"query": "What is the largest mammal in the world?" "response": "The largest mammal in the world is the blue whale."}
    {"query": "How does photosynthesis work?" "response": "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."}
    {"query": "What is the theory of relativity?" "response": "The theory of relativity developed by Albert Einstein describes the laws of physics in relation to observers in different frames of reference."}
    {"query": "Who wrote 'To Kill a Mockingbird'?" "response": "Harper Lee wrote 'To Kill a Mockingbird'."}
    ```

## Candidate Model Inference

After preparing the dataset you can use EvalScope's `run_task` method to perform inference with the candidate models and obtain their outputs for subsequent battles.

Below is an example of how to configure inference tasks for three candidate models: `Qwen2.5-0.5B-Instruct` `Qwen2.5-7B-Instruct` and `Qwen2.5-72B-Instruct` using the same configuration for inference.

Run the following code:
```python
import os
from evalscope import TaskConfig run_task
from evalscope.constants import EvalType

models = ['qwen2.5-72b-instruct' 'qwen2.5-7b-instruct' 'qwen2.5-0.5b-instruct']

task_list = [TaskConfig(
    model=model
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=os.getenv('DASHSCOPE_API_KEY')
    eval_type=EvalType.SERVICE
    datasets=[
        'general_qa'
    ]
    dataset_args={
        'general_qa': {
            'dataset_id': 'custom_eval/text/qa'
            'subset_list': [
                'arena'
                'example'
            ]
        }
    }
    eval_batch_size=10
    generation_config={
        'temperature': 0
        'n': 1
        'max_tokens': 4096
    }) for model in models]

run_task(task_cfg=task_list)
```

<details><summary>Click to view inference results</summary>

Since the `arena` subset does not have reference answers no evaluation metrics are available for this subset. The `example` subset has reference answers so evaluation metrics will be output.
```text
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric          | Subset   |   Num |   Score | Cat.0   |
+=======================+============+=================+==========+=======+=========+=========+
| qwen2.5-0.5b-instruct | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-R       | example  |    12 |  0.8611 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-P       | example  |    12 |  0.1341 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-F       | example  |    12 |  0.1983 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-R       | example  |    12 |  0.55   | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-P       | example  |    12 |  0.0404 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-F       | example  |    12 |  0.0716 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-R       | example  |    12 |  0.8611 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-P       | example  |    12 |  0.1193 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-F       | example  |    12 |  0.1754 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-1          | example  |    12 |  0.1192 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-2          | example  |    12 |  0.0403 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-3          | example  |    12 |  0.0135 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-4          | example  |    12 |  0.0079 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-P       | example  |    12 |  0.1149 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-F       | example  |    12 |  0.1612 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-R       | example  |    12 |  0.6833 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-P       | example  |    12 |  0.0813 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-F       | example  |    12 |  0.1027 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-P       | example  |    12 |  0.101  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-F       | example  |    12 |  0.1361 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-1          | example  |    12 |  0.1009 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-2          | example  |    12 |  0.0807 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-3          | example  |    12 |  0.0625 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-4          | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-P       | example  |    12 |  0.104  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-F       | example  |    12 |  0.1418 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-R       | example  |    12 |  0.7    | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-P       | example  |    12 |  0.078  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-F       | example  |    12 |  0.0964 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-P       | example  |    12 |  0.0942 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-F       | example  |    12 |  0.1235 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-1          | example  |    12 |  0.0939 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-2          | example  |    12 |  0.0777 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-3          | example  |    12 |  0.0625 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-4          | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
```
</details>

## Candidate Model Battles

Next you can use EvalScope's `general_arena` method to conduct battles among candidate models and get their win rates and rankings on each subset. To achieve robust automatic battles you need to configure an LLM as the judge that compares the outputs of models.

During evaluation EvalScope will automatically parse the public evaluation set of candidate models use the judge model to compare the output of each candidate model with the baseline and determine which is better (to avoid model bias outputs are swapped for two rounds per comparison). The judge model's outputs are parsed as win draw or loss and each candidate model's **Elo score** and **win rate** are calculated.

Run the following code:
```python
import os
from evalscope import TaskConfig run_task

task_cfg = TaskConfig(
    model_id='Arena'  # Model ID is 'Arena'; you can omit specifying model ID
    datasets=[
        'general_arena'  # Must be 'general_arena' indicating arena mode
    ]
    dataset_args={
        'general_arena': {
            # 'system_prompt': 'xxx' # Optional: customize the judge model's system prompt here
            # 'prompt_template': 'xxx' # Optional: customize the judge model's prompt template here
            'extra_params':{
                # Configure candidate model names and corresponding report paths
                # Report paths refer to the output paths from the previous step for parsing model inference results
                'models':[
                    {
                        'name': 'qwen2.5-0.5b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-0.5b-instruct'
                    }
                    {
                        'name': 'qwen2.5-7b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-7b-instruct'
                    }
                    {
                        'name': 'qwen2.5-72b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-72b-instruct'
                    }
                ]
                # Set baseline model must be one of the candidate models
                'baseline': 'qwen2.5-7b'
            }
        }
    }
    # Configure judge model parameters
    judge_model_args={
        'model_id': 'qwen-plus'
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        'api_key': os.getenv('DASHSCOPE_API_KEY')
        'generation_config': {
            'temperature': 0.0
            'max_tokens': 8000
        }
    }
    judge_worker_num=5
    # use_cache='outputs/xxx' # Optional: to add new candidate models to existing results specify the existing results path
)

run_task(task_cfg=task_cfg)
```

<details><summary>Click to view evaluation results</summary>

```text
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Model   | Dataset       | Metric        | Subset                                     |   Num |   Score | Cat.0   |
+=========+===============+===============+============================================+=======+=========+=========+
| Arena   | general_arena | winrate       | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0185 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.5469 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.075  | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.8382 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | OVERALL                                    |    44 |  0.3617 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0185 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.3906 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.025  | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.7276 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | OVERALL                                    |    44 |  0.2826 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0909 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.6875 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.0909 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.9412 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | OVERALL                                    |    44 |  0.4469 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+ 
```
</details>


The automatically generated model leaderboard is as follows (output file located in `outputs/xxx/reports/Arena/leaderboard.txt`):

The leaderboard is sorted by win rate in descending order. As shown the `qwen2.5-72b` model performs best across all subsets with the highest win rate while the `qwen2.5-0.5b` model performs the worst.

```text
=== OVERALL LEADERBOARD ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)

=== DATASET LEADERBOARD: general_qa ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)

=== SUBSET LEADERBOARD: general_qa - example ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            54.7  (-15.6 / +14.1)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            1.8  (+0.0 / +7.2)

=== SUBSET LEADERBOARD: general_qa - arena ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            83.8  (-11.1 / +10.3)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            7.5  (-5.0 / +1.6)
```

## Visualization of Battle Results

To intuitively display the results of the battles between candidate models and the baseline EvalScope provides a visualization feature allowing you to compare the results of each candidate model against the baseline model for each sample.

Run the command below to launch the visualization interface:
```shell
evalscope app
```
Open `http://localhost:7860` in your browser to view the visualization page.

Workflow:
1. Select the latest `general_arena` evaluation report and click the "Load and View" button.
2. Click dataset details and select the battle results between your candidate model and the baseline.
3. Adjust the threshold to filter battle results (normalized scores range from 0-1; 0.5 indicates a tie scores above 0.5 indicate the candidate is better than the baseline below 0.5 means worse).

Example below: a battle between `qwen2.5-72b` and `qwen2.5-7b`. The model judged the 72b as better:

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/arena_example.jpg)


# Sandbox Environment Usage

To complete LLM code capability evaluation we need to set up an independent evaluation environment to avoid executing erroneous code in the development environment and causing unavoidable losses. Currently EvalScope has integrated the [ms-enclave](https://github.com/modelscope/ms-enclave) sandbox environment allowing users to evaluate model code capabilities in a controlled environment such as using evaluation benchmarks like HumanEval and LiveCodeBench.

The following introduces two different sandbox usage methods:

- Local usage: Set up the sandbox environment on a local machine and conduct evaluation locally requiring Docker support on the local machine;
- Remote usage: Set up the sandbox environment on a remote server and conduct evaluation through API interfaces requiring Docker support on the remote machine.

## 1. Local Usage

Use Docker to set up a sandbox environment on a local machine and conduct evaluation locally requiring Docker support on the local machine.

### Environment Setup

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in your local Python environment:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration
When running evaluations add the `use_sandbox` and `sandbox_type` parameters to automatically enable the sandbox environment. Other parameters remain the same as regular evaluations:

Here's a complete example code for model evaluation on HumanEval:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig run_task

task_config = TaskConfig(
    model='qwen-plus'
    datasets=['humaneval']
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=env.get('DASHSCOPE_API_KEY')
    eval_type='openai_api'
    eval_batch_size=5
    limit=5
    generation_config={
        'max_tokens': 4096
        'temperature': 0.0
        'seed': 42
    }
    use_sandbox=True # enable sandbox
    sandbox_type='docker' # specify sandbox type
    judge_worker_num=5 # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation EvalScope will automatically start and manage the sandbox environment ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] Local sandbox manager started
...
```

## 2. Remote Usage

Set up the sandbox environment on a remote server and conduct evaluation through API interfaces requiring Docker support on the remote machine.

### Environment Setup

You need to install and configure separately on both the remote machine and local machine.

#### Remote Machine

The environment installation on the remote machine is similar to the local usage method described above:

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in remote Python environment:

```bash
pip install evalscope[sandbox]
```

3. **Start sandbox server**: Run the following command to start the sandbox server:

```bash
ms-enclave server --host 0.0.0.0 --port 1234
```

#### Local Machine

The local machine does not need Docker installation at this point but needs to install EvalScope:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration

When running evaluations add the `use_sandbox` parameter to automatically enable the sandbox environment and specify the remote sandbox server's API address in `sandbox_manager_config`:

Complete example code is as follows:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig run_task

task_config = TaskConfig(
    model='qwen-plus'
    datasets=['humaneval']
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=env.get('DASHSCOPE_API_KEY')
    eval_type='openai_api'
    eval_batch_size=5
    limit=5
    generation_config={
        'max_tokens': 4096
        'temperature': 0.0
        'seed': 42
    }
    use_sandbox=True # enable sandbox
    sandbox_type='docker' # specify sandbox type
    sandbox_manager_config={
        'base_url': 'http://<remote_host>:1234'  # remote sandbox manager URL
    }
    judge_worker_num=5 # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation EvalScope will communicate with the remote sandbox server through API ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] HTTP sandbox manager started connected to http://<remote_host>:1234
...
```


# EvalScope Service Deployment

## Introduction

EvalScope service mode provides HTTP API-based evaluation and stress testing capabilities designed to address the following scenarios:

1. **Remote Invocation**: Support remote evaluation functionality through network without configuring complex evaluation environments locally
2. **Service Integration**: Easily integrate evaluation capabilities into existing workflows CI/CD pipelines or automated testing systems
3. **Multi-user Collaboration**: Support multiple users or systems calling the evaluation service simultaneously improving resource utilization
4. **Unified Management**: Centrally manage evaluation resources and configurations for easier maintenance and monitoring
5. **Flexible Deployment**: Can be deployed on dedicated servers or container environments decoupled from business systems

The Flask service encapsulates EvalScope's core evaluation (eval) and stress testing (perf) functionalities providing services through standard RESTful APIs making evaluation capabilities callable and integrable like other microservices.

## Features

- **Model Evaluation** (`/api/v1/eval`): Support evaluation of OpenAI API-compatible models
- **Performance Testing** (`/api/v1/perf`): Support performance benchmarking of OpenAI API-compatible models
- **Parameter Query**: Provide parameter description endpoints

## Environment Setup


### Full Installation (Recommended)

```bash
pip install evalscope[service]
```

### Development Environment Installation

```bash
# Clone repository
git clone https://github.com/modelscope/evalscope.git
cd evalscope

# Install development version with service
pip install -e '.[service]'
```

## Starting the Service

### Command Line Launch

```bash
# Use default configuration (host: 0.0.0.0 port: 9000)
evalscope service

# Custom host and port
evalscope service --host 127.0.0.1 --port 9000

# Enable debug mode
evalscope service --debug
```

### Python Code Launch

```python
from evalscope.service import run_service

# Start service
run_service(host='0.0.0.0' port=9000 debug=False)
```

## API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response Example:**
```json
{
  "status": "ok"
  "service": "evalscope"
  "timestamp": "2025-12-04T10:00:00"
}
```

### 2. Model Evaluation

```bash
POST /api/v1/eval
```

**Request Body Example:**
```json
{
  "model": "qwen-plus"
  "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
  "api_key": "your-api-key"
  "datasets": ["gsm8k" "iquiz"]
  "limit": 10
  "generation_config": {
    "temperature": 0.0
    "max_tokens": 2048
  }
}
```

**Required Parameters:**
- `model`: Model name
- `datasets`: List of datasets
- `api_url`: API endpoint URL (OpenAI-compatible)

**Optional Parameters:**
- `api_key`: API key (default: "EMPTY")
- `limit`: Evaluation sample quantity limit
- `eval_batch_size`: Batch size (default: 1)
- `generation_config`: Generation configuration
  - `temperature`: Temperature parameter (default: 0.0)
  - `max_tokens`: Maximum generation tokens (default: 2048)
  - `top_p`: Nucleus sampling parameter
  - `top_k`: Top-k sampling parameter
- `work_dir`: Output directory
- `debug`: Debug mode
- `seed`: Random seed (default: 42)

**Response Example:**
```json
{
  "status": "success"
  "message": "Evaluation completed"
  "result": {"...": "..."}
  "output_dir": "/path/to/outputs/20251204_100000"
}
```

### 3. Performance Testing

```bash
POST /api/v1/perf
```

**Request Body Example:**
```json
{
  "model": "qwen-plus"
  "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  "api": "openai"
  "api_key": "your-api-key"
  "number": 100
  "parallel": 10
  "dataset": "openqa"
  "max_tokens": 2048
  "temperature": 0.0
}
```

**Required Parameters:**
- `model`: Model name
- `url`: Complete API endpoint URL

**Optional Parameters:**
- `api`: API type (openai/dashscope/anthropic/gemini default: "openai")
- `api_key`: API key
- `number`: Total number of requests (default: 1000)
- `parallel`: Concurrency level (default: 1)
- `rate`: Requests per second limit (default: -1 unlimited)
- `dataset`: Dataset name (default: "openqa")
- `max_tokens`: Maximum generation tokens (default: 2048)
- `temperature`: Temperature parameter (default: 0.0)
- `stream`: Whether to use streaming output (default: true)
- `debug`: Debug mode

**Response Example:**
```json
{
  "status": "success"
  "message": "Performance test completed"
  "output_dir": "/path/to/outputs"
  "results": {
    "parallel_10_number_100": {
      "metrics": {"...": "..."}
      "percentiles": {"...": "..."}
    }
  }
}
```

### 4. Get Evaluation Parameter Description

```bash
GET /api/v1/eval/params
```

Returns descriptions of all parameters supported by the evaluation endpoint.

### 5. Get Performance Test Parameter Description

```bash
GET /api/v1/perf/params
```

Returns descriptions of all parameters supported by the performance test endpoint.

## Usage Examples

### Testing Evaluation Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "api_key": "your-api-key"
    "datasets": ["gsm8k"]
    "limit": 5
  }'
```

### Testing Performance Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/perf \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    "api": "openai"
    "number": 50
    "parallel": 5
  }'
```

### Using Python requests

```python
import requests

# Evaluation request
eval_response = requests.post(
    'http://localhost:9000/api/v1/eval'
    json={
        'model': 'qwen-plus'
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        'api_key': 'your-api-key'
        'datasets': ['gsm8k' 'iquiz']
        'limit': 10
        'generation_config': {
            'temperature': 0.0
            'max_tokens': 2048
        }
    }
)
print(eval_response.json())

# Performance test request
perf_response = requests.post(
    'http://localhost:9000/api/v1/perf'
    json={
        'model': 'qwen-plus'
        'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        'api': 'openai'
        'number': 100
        'parallel': 10
        'dataset': 'openqa'
    }
)
print(perf_response.json())
```

## Important Notes

1. **OpenAI API-Compatible Models Only**: This service is designed specifically for OpenAI API-compatible models
2. **Long-Running Tasks**: Evaluation and performance testing tasks may take considerable time. We recommend setting appropriate HTTP timeout values on the client side as the API calls are synchronous and will block until completion.
3. **Output Directory**: Evaluation results are saved in the configured `work_dir` default is `outputs/`
4. **Error Handling**: The service returns detailed error messages and stack traces (in debug mode)
5. **Resource Management**: Pay attention to concurrency settings during stress testing to avoid server overload

## Error Codes

- `400`: Invalid request parameters
- `404`: Endpoint not found
- `500`: Internal server error

## Example Scenarios

### Scenario 1: Quick Evaluation of Qwen Model

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "api_key": "sk-..."
    "datasets": ["gsm8k"]
    "limit": 100
  }'
```

### Scenario 2: Stress Testing Locally Deployed Model

```bash
curl -X POST http://localhost:9000/api/v1/perf \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5"
    "url": "http://localhost:8000/v1/chat/completions"
    "api": "openai"
    "number": 1000
    "parallel": 20
    "max_tokens": 2048
  }'
```

### Scenario 3: Multi-Dataset Evaluation

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "datasets": ["gsm8k" "iquiz" "ceval"]
    "limit": 50
    "eval_batch_size": 4
  }'
```

<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  Chinese Documentation</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documentation</a>
<p>


> ⭐ If you like this project please click the "Star" button in the upper right corner to support us. Your support is our motivation to move forward!

## 📝 Introduction

EvalScope is a powerful and easily extensible model evaluation framework created by the [ModelScope Community](https://modelscope.cn/) aiming to provide a one-stop evaluation solution for large model developers.

Whether you want to evaluate the general capabilities of models conduct multi-model performance comparisons or need to stress test models EvalScope can meet your needs.

## ✨ Key Features

- **📚 Comprehensive Evaluation Benchmarks**: Built-in multiple industry-recognized evaluation benchmarks including MMLU C-Eval GSM8K and more.
- **🧩 Multi-modal and Multi-domain Support**: Supports evaluation of various model types including Large Language Models (LLM) Vision Language Models (VLM) Embedding Reranker AIGC and more.
- **🚀 Multi-backend Integration**: Seamlessly integrates multiple evaluation backends including OpenCompass VLMEvalKit RAGEval to meet different evaluation needs.
- **⚡ Inference Performance Testing**: Provides powerful model service stress testing tools supporting multiple performance metrics such as TTFT TPOT.
- **📊 Interactive Reports**: Provides WebUI visualization interface supporting multi-dimensional model comparison report overview and detailed inspection.
- **⚔️ Arena Mode**: Supports multi-model battles (Pairwise Battle) intuitively ranking and evaluating models.
- **🔧 Highly Extensible**: Developers can easily add custom datasets models and evaluation metrics.

<details><summary>🏛️ Overall Architecture</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope Overall Architecture.
</p>

1.  **Input Layer**
    - **Model Sources**: API models (OpenAI API) Local models (ModelScope)
    - **Datasets**: Standard evaluation benchmarks (MMLU/GSM8k etc.) Custom data (MCQ/QA)

2.  **Core Functions**
    - **Multi-backend Evaluation**: Native backend OpenCompass MTEB VLMEvalKit RAGAS
    - **Performance Monitoring**: Supports multiple model service APIs and data formats tracking TTFT/TPOP and other metrics
    - **Tool Extensions**: Integrates Tool-Bench Needle-in-a-Haystack etc.

3.  **Output Layer**
    - **Structured Reports**: Supports JSON Table Logs
    - **Visualization Platform**: Supports Gradio Wandb SwanLab

</details>

## 🎉 What's New

> [!IMPORTANT]
> **Version 1.0 Refactoring**
>
> Version 1.0 introduces a major overhaul of the evaluation framework establishing a new more modular and extensible API layer under `evalscope/api`. Key improvements include standardized data models for benchmarks samples and results; a registry-based design for components such as benchmarks and metrics; and a rewritten core evaluator that orchestrates the new architecture. Existing benchmark adapters have been migrated to this API resulting in cleaner more consistent and easier-to-maintain implementations.

- 🔥 **[2025.12.02]** Added support for custom multimodal VQA evaluation; refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html). Added support for visualizing model service stress testing in ClearML; refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#clearml).
- 🔥 **[2025.11.26]** Added support for OpenAI-MRCR GSM8K-V MGSM MicroVQA IFBench SciCode benchmarks.
- 🔥 **[2025.11.18]** Added support for custom Function-Call (tool invocation) datasets to test whether models can timely and correctly call tools. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#function-calling-format-fc).
- 🔥 **[2025.11.14]** Added support for SWE-bench_Verified SWE-bench_Lite SWE-bench_Verified_mini code evaluation benchmarks. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html).
- 🔥 **[2025.11.12]** Added `pass@k` `vote@k` `pass^k` and other metric aggregation methods; added support for multimodal evaluation benchmarks such as A_OKVQA CMMU ScienceQA V*Bench.
- 🔥 **[2025.11.07]** Added support for τ²-bench an extended and enhanced version of τ-bench that includes a series of code fixes and adds telecom domain troubleshooting scenarios. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/tau2_bench.html).
- 🔥 **[2025.10.30]** Added support for BFCL-v4 enabling evaluation of agent capabilities including web search and long-term memory. See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v4.html).
- 🔥 **[2025.10.27]** Added support for LogiQA HaluEval MathQA MRI-QA PIQA QASC CommonsenseQA and other evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.26]** Added support for Conll-2003 CrossNER Copious GeniaNER HarveyNER MIT-Movie-Trivia MIT-Restaurant OntoNotes5 WNUT2017 and other Named Entity Recognition evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.21]** Optimized sandbox environment usage in code evaluation supporting both local and remote operation modes. For details refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html).
- 🔥 **[2025.10.20]** Added support for evaluation benchmarks including PolyMath SimpleVQA MathVerse MathVision AA-LCR; optimized evalscope perf performance to align with vLLM Bench. For details refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/vs_vllm_bench.html).
- 🔥 **[2025.10.14]** Added support for OCRBench OCRBench-v2 DocVQA InfoVQA ChartQA and BLINK multimodal image-text evaluation benchmarks.
- 🔥 **[2025.09.22]** Code evaluation benchmarks (HumanEval LiveCodeBench) now support running in a sandbox environment. To use this feature please install [ms-enclave](https://github.com/modelscope/ms-enclave) first.
- 🔥 **[2025.09.19]** Added support for multimodal image-text evaluation benchmarks including RealWorldQA AI2D MMStar MMBench and OmniBench as well as pure text evaluation benchmarks such as Multi-IF HealthBench and AMC.
- 🔥 **[2025.09.05]** Added support for vision-language multimodal model evaluation tasks such as MathVista and MMMU. For more supported datasets please [refer to the documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/vlm.html).
- 🔥 **[2025.09.04]** Added support for image editing task evaluation including the [GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) benchmark. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/image_edit.html).
- 🔥 **[2025.08.22]** Version 1.0 Refactoring. Break changes please [refer to](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#switching-to-version-v1-0).
<details><summary>More</summary>

- 🔥 **[2025.07.18]** The model stress testing now supports randomly generating image-text data for multimodal model evaluation. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#id4).
- 🔥 **[2025.07.16]** Support for [τ-bench](https://github.com/sierra-research/tau-bench) has been added enabling the evaluation of AI Agent performance and reliability in real-world scenarios involving dynamic user and tool interactions. For usage instructions please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html#bench).
- 🔥 **[2025.07.14]** Support for "Humanity's Last Exam" ([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle)) a highly challenging evaluation benchmark. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam).
- 🔥 **[2025.07.03]** Refactored Arena Mode: now supports custom model battles outputs a model leaderboard and provides battle result visualization. See [reference](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html) for details.
- 🔥 **[2025.06.28]** Optimized custom dataset evaluation: now supports evaluation without reference answers. Enhanced LLM judge usage with built-in modes for "scoring directly without reference answers" and "checking answer consistency with reference answers". See [reference](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#qa) for details.
- 🔥 **[2025.06.19]** Added support for the [BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3) benchmark designed to evaluate model function-calling capabilities across various scenarios. For more information refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html).
- 🔥 **[2025.06.02]** Added support for the Needle-in-a-Haystack test. Simply specify `needle_haystack` to conduct the test and a corresponding heatmap will be generated in the `outputs/reports` folder providing a visual representation of the model's performance. Refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/needle_haystack.html) for more details.
- 🔥 **[2025.05.29]** Added support for two long document evaluation benchmarks: [DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary) and [FRAMES](https://modelscope.cn/datasets/iic/frames/summary). For usage guidelines please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html).
- 🔥 **[2025.05.16]** Model service performance stress testing now supports setting various levels of concurrency and outputs a performance test report. [Reference example](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html#id3).
- 🔥 **[2025.05.13]** Added support for the [ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static) dataset to evaluate model's tool-calling capabilities. Refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html) for usage instructions. Also added support for the [DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview) and [Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val) benchmarks to assess the reasoning capabilities of models.
- 🔥 **[2025.04.29]** Added Qwen3 Evaluation Best Practices [welcome to read 📖](https://evalscope.readthedocs.io/en/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** Support for text-to-image evaluation: Supports 8 metrics including MPS HPSv2.1Score etc. and evaluation benchmarks such as EvalMuse GenAI-Bench. Refer to the [user documentation](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/t2i.html) for more details.
- 🔥 **[2025.04.10]** Model service stress testing tool now supports the `/v1/completions` endpoint (the default endpoint for vLLM benchmarking)
- 🔥 **[2025.04.08]** Support for evaluating embedding model services compatible with the OpenAI API has been added. For more details check the [user guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters).
- 🔥 **[2025.03.27]** Added support for [AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview) and [ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary) evaluation benchmarks. For usage notes please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** The model inference service stress testing now supports generating prompts of specified length using random values. Refer to the [user guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#using-the-random-dataset) for more details.
- 🔥 **[2025.03.13]** Added support for the [LiveCodeBench](https://www.modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary) code evaluation benchmark which can be used by specifying `live_code_bench`. Supports evaluating QwQ-32B on LiveCodeBench refer to the [best practices](https://evalscope.readthedocs.io/en/latest/best_practice/eval_qwq.html).
- 🔥 **[2025.03.11]** Added support for the [SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary) and [Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary) evaluation benchmarks. These are used to assess the factual accuracy of models and you can specify `simple_qa` and `chinese_simpleqa` for use. Support for specifying a judge model is also available. For more details refer to the [relevant parameter documentation](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).
- 🔥 **[2025.03.07]** Added support for the [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/summary) model evaluate the model's reasoning ability and reasoning efficiency refer to [📖 Best Practices for QwQ-32B Evaluation](https://evalscope.readthedocs.io/en/latest/best_practice/eval_qwq.html) for more details.
- 🔥 **[2025.03.04]** Added support for the [SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary) dataset which covers 13 categories 72 first-level disciplines and 285 second-level disciplines totaling 26529 questions. You can use it by specifying `super_gpqa`.
- 🔥 **[2025.03.03]** Added support for evaluating the IQ and EQ of models. Refer to [📖 Best Practices for IQ and EQ Evaluation](https://evalscope.readthedocs.io/en/latest/best_practice/iquiz.html) to find out how smart your AI is!
- 🔥 **[2025.02.27]** Added support for evaluating the reasoning efficiency of models. Refer to [📖 Best Practices for Evaluating Thinking Efficiency](https://evalscope.readthedocs.io/en/latest/best_practice/think_eval.html). This implementation is inspired by the works [Overthinking](https://doi.org/10.48550/arXiv.2412.21187) and [Underthinking](https://doi.org/10.48550/arXiv.2501.18585).
- 🔥 **[2025.02.25]** Added support for two model inference-related evaluation benchmarks: [MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR) and [ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary). To use them simply specify `musr` and `process_bench` respectively in the datasets parameter.
- 🔥 **[2025.02.18]** Supports the AIME25 dataset which contains 15 questions (Grok3 scored 93 on this dataset).
- 🔥 **[2025.02.13]** Added support for evaluating DeepSeek distilled models including AIME24 MATH-500 and GPQA-Diamond datasets，refer to [best practice](https://evalscope.readthedocs.io/en/latest/best_practice/deepseek_r1_distill.html); Added support for specifying the `eval_batch_size` parameter to accelerate model evaluation.
- 🔥 **[2025.01.20]** Support for visualizing evaluation results including single model evaluation results and multi-model comparison refer to the [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html) for more details; Added [`iquiz`](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary) evaluation example evaluating the IQ and EQ of the model.
- 🔥 **[2025.01.07]** Native backend: Support for model API evaluation is now available. Refer to the [📖 Model API Evaluation Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#api) for more details. Additionally support for the `ifeval` evaluation benchmark has been added.
- 🔥🔥 **[2024.12.31]** Support for adding benchmark evaluations refer to the [📖 Benchmark Evaluation Addition Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html); support for custom mixed dataset evaluations allowing for more comprehensive model evaluations with less data refer to the [📖 Mixed Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/collection/index.html).
- 🔥 **[2024.12.13]** Model evaluation optimization: no need to pass the `--template-type` parameter anymore; supports starting evaluation with `evalscope eval --args`. Refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html) for more details.
- 🔥 **[2024.11.26]** The model inference service performance evaluator has been completely refactored: it now supports local inference service startup and Speed Benchmark; asynchronous call error handling has been optimized. For more details refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).
- 🔥 **[2024.10.31]** The best practice for evaluating Multimodal-RAG has been updated please check the [📖 Blog](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag) for more details.
- 🔥 **[2024.10.23]** Supports multimodal RAG evaluation including the assessment of image-text retrieval using [CLIP_Benchmark](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/clip_benchmark.html) and extends [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html) to support end-to-end multimodal metrics evaluation.
- 🔥 **[2024.10.8]** Support for RAG evaluation including independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html) as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html).
- 🔥 **[2024.09.18]** Our documentation has been updated to include a blog module featuring some technical research and discussions related to evaluations. We invite you to [📖 read it](https://evalscope.readthedocs.io/en/refact_readme/blog/index.html).
- 🔥 **[2024.09.12]** Support for LongWriter evaluation which supports 10000+ word generation. You can use the benchmark [LongBench-Write](evalscope/third_party/longbench_write/README.md) to measure the long output quality as well as the output length.
- 🔥 **[2024.08.30]** Support for custom dataset evaluations including text datasets and multimodal image-text datasets.
- 🔥 **[2024.08.20]** Updated the official documentation including getting started guides best practices and FAQs. Feel free to [📖read it here](https://evalscope.readthedocs.io/en/latest/)!
- 🔥 **[2024.08.09]** Simplified the installation process allowing for pypi installation of vlmeval dependencies; optimized the multimodal model evaluation experience achieving up to 10x acceleration based on the OpenAI API evaluation chain.
- 🔥 **[2024.07.31]** Important change: The package name `llmuses` has been changed to `evalscope`. Please update your code accordingly.
- 🔥 **[2024.07.26]** Support for **VLMEvalKit** as a third-party evaluation framework to initiate multimodal model evaluation tasks.
- 🔥 **[2024.06.29]** Support for **OpenCompass** as a third-party evaluation framework which we have encapsulated at a higher level supporting pip installation and simplifying evaluation task configuration.
- 🔥 **[2024.06.13]** EvalScope seamlessly integrates with the fine-tuning framework SWIFT providing full-chain support from LLM training to evaluation.
- 🔥 **[2024.06.13]** Integrated the Agent evaluation dataset ToolBench.

</details>

## ❤️ Community & Support

Welcome to join our community to communicate with other developers and get help.

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  WeChat Group | DingTalk Group
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">



## 🛠️ Environment Setup

We recommend using `conda` to create a virtual environment and install with `pip`.

1.  **Create and Activate Conda Environment** (Python 3.10 recommended)
    ```shell
    conda create -n evalscope python=3.10
    conda activate evalscope
    ```

2.  **Install EvalScope**

    - **Method 1: Install via PyPI (Recommended)**
      ```shell
      pip install evalscope
      ```

    - **Method 2: Install from Source (For Development)**
      ```shell
      git clone https://github.com/modelscope/evalscope.git
      cd evalscope
      pip install -e .
      ```

3.  **Install Additional Dependencies** (Optional)
    Install corresponding feature extensions according to your needs:
    ```shell
    # Performance testing
    pip install 'evalscope[perf]'

    # Visualization App
    pip install 'evalscope[app]'

    # Other evaluation backends
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'

    # Install all dependencies
    pip install 'evalscope[all]'
    ```
    > If you installed from source please replace `evalscope` with `.` for example `pip install '.[perf]'`.

> [!NOTE]
> This project was formerly known as `llmuses`. If you need to use `v0.4.3` or earlier versions please run `pip install llmuses<=0.4.3` and use `from llmuses import ...` for imports.


## 🚀 Quick Start

You can start evaluation tasks in two ways: **command line** or **Python code**.

### Method 1. Using Command Line

Execute the `evalscope eval` command in any path to start evaluation. The following command will evaluate the `Qwen/Qwen2.5-0.5B-Instruct` model on `gsm8k` and `arc` datasets taking only 5 samples from each dataset.

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### Method 2. Using Python Code

Use the `run_task` function and `TaskConfig` object to configure and start evaluation tasks.

```python
from evalscope import run_task TaskConfig

# Configure evaluation task
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct'
    datasets=['gsm8k' 'arc']
    limit=5
)

# Start evaluation
run_task(task_cfg)
```

<details><summary><b>💡 Tip:</b> `run_task` also supports dictionaries YAML or JSON files as configuration.</summary>

**Using Python Dictionary**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct'
    'datasets': ['gsm8k' 'arc']
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**Using YAML File** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### Output Results
After evaluation completion you will see a report in the terminal in the following format:
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 📈 Advanced Usage

### Custom Evaluation Parameters

You can fine-tune model loading inference and dataset configuration through command line parameters.

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master" "precision": "torch.float16" "device_map": "auto"}' \
 --generation-config '{"do_sample":true"temperature":0.6"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0 "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: Model loading parameters such as `revision` `precision` etc.
- `--generation-config`: Model generation parameters such as `temperature` `max_tokens` etc.
- `--dataset-args`: Dataset configuration parameters such as `few_shot_num` etc.

For details please refer to [📖 Complete Parameter Guide](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).

### Evaluating Online Model APIs

EvalScope supports evaluating model services deployed via APIs (such as services deployed with vLLM). Simply specify the service address and API Key.

1.  **Start Model Service** (using vLLM as example)
    ```shell
    export VLLM_USE_MODELSCOPE=True
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen2.5 \
      --port 8801
    ```

2.  **Run Evaluation**
    ```shell
    evalscope eval \
     --model qwen2.5 \
     --eval-type openai_api \
     --api-url http://127.0.0.1:8801/v1 \
     --api-key EMPTY \
     --datasets gsm8k \
     --limit 10
    ```

### ⚔️ Arena Mode

Arena mode evaluates model performance through pairwise battles between models providing win rates and rankings perfect for horizontal comparison of multiple models.

```text
# Example evaluation results
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
For details please refer to [📖 Arena Mode Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html).

### 🖊️ Custom Dataset Evaluation

EvalScope allows you to easily add and evaluate your own datasets. For details please refer to [📖 Custom Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html).


## 🧪 Other Evaluation Backends
EvalScope supports launching evaluation tasks through third-party evaluation frameworks (we call them "backends") to meet diverse evaluation needs.

- **Native**: EvalScope's default evaluation framework with comprehensive functionality.
- **OpenCompass**: Focuses on text-only evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: Focuses on multi-modal evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: Focuses on RAG evaluation supporting Embedding and Reranker models. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/index.html)
- **Third-party Evaluation Tools**: Supports evaluation tasks like [ToolBench](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html).

## ⚡ Inference Performance Evaluation Tool
EvalScope provides a powerful stress testing tool for evaluating the performance of large language model services.

- **Key Metrics**: Supports throughput (Tokens/s) first token latency (TTFT) token generation latency (TPOT) etc.
- **Result Recording**: Supports recording results to `wandb` and `swanlab`.
- **Speed Benchmarks**: Can generate speed benchmark results similar to official reports.

For details please refer to [📖 Performance Testing Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).

Example output is shown below:
<p align="center">
    <img src="docs/en/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>


## 📊 Visualizing Evaluation Results

EvalScope provides a Gradio-based WebUI for interactive analysis and comparison of evaluation results.

1.  **Install Dependencies**
    ```bash
    pip install 'evalscope[app]'
    ```

2.  **Start Service**
    ```bash
    evalscope app
    ```
    Visit `http://127.0.0.1:7861` to open the visualization interface.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/setting.png" alt="Setting" style="width: 85%;" />
      <p>Settings Interface</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/model_compare.png" alt="Model Compare" style="width: 100%;" />
      <p>Model Comparison</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/report_overview.png" alt="Report Overview" style="width: 100%;" />
      <p>Report Overview</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/report_details.png" alt="Report Details" style="width: 85%;" />
      <p>Report Details</p>
    </td>
  </tr>
</table>

For details please refer to [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html).

## 👷‍♂️ Contributing

We welcome any contributions from the community! If you want to add new evaluation benchmarks models or features please refer to our [Contributing Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html).

Thanks to all developers who have contributed to EvalScope!

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>


## 📚 Citation

If you use EvalScope in your research please cite our work:
```bibtex
@misc{evalscope_2024
    title={{EvalScope}: Evaluation Framework for Large Models}
    author={ModelScope Team}
    year={2024}
    url={https://github.com/modelscope/evalscope}
}
```


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)

<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  中文 &nbsp ｜ &nbsp <a href="evalscope.md">English</a> &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/zh-cn/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documents</a>
<p>


> ⭐ 如果你喜欢这个项目，请点击右上角的 "Star" 按钮支持我们。你的支持是我们前进的动力！

## 📝 简介

EvalScope 是由[魔搭社区](https://modelscope.cn/)打造的一款功能强大、易于扩展的模型评测框架，旨在为大模型开发者提供一站式评测解决方案。

无论您是想评估模型的通用能力、进行多模型性能对比，还是需要对模型进行压力测试，EvalScope 都能满足您的需求。

## ✨ 主要特性

- **📚 全面的评测基准**: 内置 MMLU C-Eval GSM8K 等多个业界公认的评测基准。
- **🧩 多模态与多领域支持**: 支持大语言模型 (LLM)、多模态 (VLM)、Embedding、Reranker、AIGC 等多种模型的评测。
- **🚀 多后端集成**: 无缝集成 OpenCompass VLMEvalKit RAGEval 等多种评测后端，满足不同评测需求。
- **⚡ 推理性能测试**: 提供强大的模型服务压力测试工具，支持 TTFT TPOT 等多项性能指标。
- **📊 交互式报告**: 提供 WebUI 可视化界面，支持多维度模型对比、报告概览和详情查阅。
- **⚔️ 竞技场模式**: 支持多模型对战 (Pairwise Battle)，直观地对模型进行排名和评估。
- **🔧 高度可扩展**: 开发者可以轻松添加自定义数据集、模型和评测指标。

<details><summary>🏛️ 整体架构</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope 整体架构图.
</p>

1.  **输入层**
    - **模型来源**: API模型（OpenAI API）、本地模型（ModelScope）
    - **数据集**: 标准评测基准（MMLU/GSM8k等）、自定义数据（MCQ/QA）

2.  **核心功能**
    - **多后端评估**: 原生后端、OpenCompass、MTEB、VLMEvalKit、RAGAS
    - **性能监控**: 支持多种模型服务 API 和数据格式，追踪 TTFT/TPOP 等指标
    - **工具扩展**: 集成 Tool-Bench Needle-in-a-Haystack 等

3.  **输出层**
    - **结构化报告**: 支持 JSON Table Logs
    - **可视化平台**: 支持 Gradio Wandb SwanLab

</details>

## 🎉 内容更新

> [!IMPORTANT]
> **版本 1.0 重构**
>
> 版本 1.0 对评测框架进行了重大重构，在 `evalscope/api` 下建立了全新的、更模块化且易扩展的 API 层。主要改进包括：为基准、样本和结果引入了标准化数据模型；对基准和指标等组件采用注册表式设计；并重写了核心评测器以协同新架构。现有的基准已迁移到这一 API，实现更加简洁、一致且易于维护。

- 🔥 **[2025.12.02]** 支持自定义多模态VQA评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html) ；支持模型服务压测在 ClearML 上可视化，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#clearml)。
- 🔥 **[2025.11.26]** 新增支持 OpenAI-MRCR、GSM8K-V、MGSM、MicroVQA、IFBench、SciCode 评测基准。
- 🔥 **[2025.11.18]** 支持自定义 Function-Call（工具调用）数据集，来测试模型能否适时并正确调用工具，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#fc)
- 🔥 **[2025.11.14]** 新增支持SWE-bench_Verified SWE-bench_Lite SWE-bench_Verified_mini 代码评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)。
- 🔥 **[2025.11.12]** 新增`pass@k`、`vote@k`、`pass^k`等指标聚合方法；新增支持A_OKVQA CMMU ScienceQ V*Bench等多模态评测基准。
- 🔥 **[2025.11.07]** 新增支持τ²-bench，是 τ-bench 的扩展与增强版本，包含一系列代码修复，并新增了电信（telecom）领域的故障排查场景，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau2_bench.html)。
- 🔥 **[2025.10.30]** 新增支持BFCL-v4，支持agent的网络搜索和长期记忆能力的评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v4.html)。
- 🔥 **[2025.10.27]** 新增支持LogiQA HaluEval MathQA MRI-QA PIQA QASC CommonsenseQA等评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.26]** 新增支持Conll-2003 CrossNER Copious GeniaNER HarveyNER MIT-Movie-Trivia MIT-Restaurant OntoNotes5 WNUT2017 等命名实体识别评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.21]** 优化代码评测中的沙箱环境使用，支持在本地和远程两种模式下运行，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 🔥 **[2025.10.20]** 新增支持PolyMath SimpleVQA MathVerse MathVision AA-LCR 等评测基准；优化evalscope perf表现，对齐vLLM Bench，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/vs_vllm_bench.html)。
- 🔥 **[2025.10.14]** 新增支持OCRBench OCRBench-v2 DocVQA InfoVQA ChartQA BLINK 等图文多模态评测基准。
- 🔥 **[2025.09.22]** 代码评测基准(HumanEval LiveCodeBench)支持在沙箱环境中运行，要使用该功能需先安装[ms-enclave](https://github.com/modelscope/ms-enclave)。
- 🔥 **[2025.09.19]** 新增支持RealWorldQA、AI2D、MMStar、MMBench、OmniBench等图文多模态评测基准，和Multi-IF、HealthBench、AMC等纯文本评测基准。
- 🔥 **[2025.09.05]** 支持视觉-语言多模态大模型的评测任务，例如：MathVista、MMMU，更多支持数据集请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/vlm.html)。
- 🔥 **[2025.09.04]** 支持图像编辑任务评测，支持[GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) 评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/image_edit.html)。
- 🔥 **[2025.08.22]** Version 1.0 重构，不兼容的更新请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#v1-0)。
<details> <summary>更多</summary>

- 🔥 **[2025.07.18]** 模型压测支持随机生成图文数据，用于多模态模型压测，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#id4)。
- 🔥 **[2025.07.16]** 支持[τ-bench](https://github.com/sierra-research/tau-bench)，用于评估 AI Agent在动态用户和工具交互的实际环境中的性能和可靠性，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#bench)。
- 🔥 **[2025.07.14]** 支持“人类最后的考试”([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle))，这一高难度评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam)。
- 🔥 **[2025.07.03]** 重构了竞技场模式，支持自定义模型对战，输出模型排行榜，以及对战结果可视化，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。
- 🔥 **[2025.06.28]** 优化自定义数据集评测，支持无参考答案评测；优化LLM裁判使用，预置“无参考答案直接打分” 和 “判断答案是否与参考答案一致”两种模式，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa)
- 🔥 **[2025.06.19]** 新增支持[BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3)评测基准，用于评测模型在多种场景下的函数调用能力，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)。
- 🔥 **[2025.06.02]** 新增支持大海捞针测试（Needle-in-a-Haystack），指定`needle_haystack`即可进行测试，并在`outputs/reports`文件夹下生成对应的heatmap，直观展现模型性能，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)。
- 🔥 **[2025.05.29]** 新增支持[DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)和[FRAMES](https://modelscope.cn/datasets/iic/frames/summary)两个长文档评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.05.16]** 模型服务性能压测支持设置多种并发，并输出性能压测报告，[参考示例](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id3)。
- 🔥 **[2025.05.13]** 新增支持[ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static)数据集，评测模型的工具调用能力，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)；支持[DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview)和[Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val)评测基准，评测模型的推理能力。
- 🔥 **[2025.04.29]** 新增Qwen3评测最佳实践，[欢迎阅读📖](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** 支持文生图评测：支持MPS、HPSv2.1Score等8个指标，支持EvalMuse、GenAI-Bench等评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/t2i.html)
- 🔥 **[2025.04.10]** 模型服务压测工具支持`/v1/completions`端点（也是vLLM基准测试的默认端点）
- 🔥 **[2025.04.08]** 支持OpenAI API兼容的Embedding模型服务评测，查看[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters)
- 🔥 **[2025.03.27]** 新增支持[AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview)和[ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** 模型推理服务压测支持random生成指定范围长度的prompt，参考[使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random)
- 🔥 **[2025.03.13]** 新增支持[LiveCodeBench](https://www.modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary)代码评测基准，指定`live_code_bench`即可使用；支持QwQ-32B 在LiveCodeBench上评测，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.11]** 新增支持[SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)和[Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)评测基准，用与评测模型的事实正确性，指定`simple_qa`和`chinese_simpleqa`使用。同时支持指定裁判模型，参考[相关参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。
- 🔥 **[2025.03.07]** 新增QwQ-32B模型评测最佳实践，评测了模型的推理能力以及推理效率，参考[📖QwQ-32B模型评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.04]** 新增支持[SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)数据集，其覆盖 13 个门类、72 个一级学科和 285 个二级学科，共 26529 个问题，指定`super_gpqa`即可使用。
- 🔥 **[2025.03.03]** 新增支持评测模型的智商和情商，参考[📖智商和情商评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/iquiz.html)，来测测你家的AI有多聪明？
- 🔥 **[2025.02.27]** 新增支持评测推理模型的思考效率，参考[📖思考效率评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/think_eval.html)，该实现参考了[Overthinking](https://doi.org/10.48550/arXiv.2412.21187) 和 [Underthinking](https://doi.org/10.48550/arXiv.2501.18585)两篇工作。
- 🔥 **[2025.02.25]** 新增支持[MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR)和[ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary)两个模型推理相关评测基准，datasets分别指定`musr`和`process_bench`即可使用。
- 🔥 **[2025.02.18]** 支持AIME25数据集，包含15道题目（Grok3 在该数据集上得分为93分）
- 🔥 **[2025.02.13]** 支持DeepSeek蒸馏模型评测，包括AIME24 MATH-500 GPQA-Diamond数据集，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/deepseek_r1_distill.html)；支持指定`eval_batch_size`参数，加速模型评测
- 🔥 **[2025.01.20]** 支持可视化评测结果，包括单模型评测结果和多模型评测结果对比，参考[📖可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)；新增[`iquiz`](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)评测样例，评测模型的IQ和EQ。
- 🔥 **[2025.01.07]** Native backend: 支持模型API评测，参考[📖模型API评测指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#api)；新增支持`ifeval`评测基准。
- 🔥🔥 **[2024.12.31]** 支持基准评测添加，参考[📖基准评测添加指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)；支持自定义混合数据集评测，用更少的数据，更全面的评测模型，参考[📖混合数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html)
- 🔥 **[2024.12.13]** 模型评测优化，不再需要传递`--template-type`参数；支持`evalscope eval --args`启动评测，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html)
- 🔥 **[2024.11.26]** 模型推理压测工具重构完成：支持本地启动推理服务、支持Speed Benchmark；优化异步调用错误处理，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)
- 🔥 **[2024.10.31]** 多模态RAG评测最佳实践发布，参考[📖博客](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag)
- 🔥 **[2024.10.23]** 支持多模态RAG评测，包括[CLIP_Benchmark](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/clip_benchmark.html)评测图文检索器，以及扩展了[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)以支持端到端多模态指标评测。
- 🔥 **[2024.10.8]** 支持RAG评测，包括使用[MTEB/CMTEB](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html)进行embedding模型和reranker的独立评测，以及使用[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)进行端到端评测。
- 🔥 **[2024.09.18]** 我们的文档增加了博客模块，包含一些评测相关的技术调研和分享，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/blog/index.html)
- 🔥 **[2024.09.12]** 支持 LongWriter 评测，您可以使用基准测试 [LongBench-Write](evalscope/third_party/longbench_write/README.md) 来评测长输出的质量以及输出长度。
- 🔥 **[2024.08.30]** 支持自定义数据集评测，包括文本数据集和多模态图文数据集。
- 🔥 **[2024.08.20]** 更新了官方文档，包括快速上手、最佳实践和常见问题等，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/)。
- 🔥 **[2024.08.09]** 简化安装方式，支持pypi安装vlmeval相关依赖；优化多模态模型评测体验，基于OpenAI API方式的评测链路，最高加速10倍。
- 🔥 **[2024.07.31]** 重要修改：`llmuses`包名修改为`evalscope`，请同步修改您的代码。
- 🔥 **[2024.07.26]** 支持**VLMEvalKit**作为第三方评测框架，发起多模态模型评测任务。
- 🔥 **[2024.06.29]** 支持**OpenCompass**作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评测任务配置。
- 🔥 **[2024.06.13]** EvalScope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 。
- 🔥 **[2024.06.13]** 接入Agent评测集ToolBench。
</details>

## ❤️ 社区与支持

欢迎加入我们的社区，与其他开发者交流并获取帮助。

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  微信群 | 钉钉群
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">



## 🛠️ 环境准备

我们推荐使用 `conda` 创建虚拟环境，并使用 `pip` 安装。

1.  **创建并激活 Conda 环境** (推荐使用 Python 3.10)
    ```shell
    conda create -n evalscope python=3.10
    conda activate evalscope
    ```

2.  **安装 EvalScope**

    - **方式一：通过 PyPI 安装 (推荐)**
      ```shell
      pip install evalscope
      ```

    - **方式二：通过源码安装 (用于开发)**
      ```shell
      git clone https://github.com/modelscope/evalscope.git
      cd evalscope
      pip install -e .
      ```

3.  **安装额外依赖** (可选)
    根据您的需求，安装相应的功能扩展：
    ```shell
    # 性能测试
    pip install 'evalscope[perf]'

    # 可视化App
    pip install 'evalscope[app]'

    # 其他评测后端
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'

    # 安装所有依赖
    pip install 'evalscope[all]'
    ```
    > 如果您通过源码安装，请将 `evalscope` 替换为 `.`，例如 `pip install '.[perf]'`。

> [!NOTE]
> 本项目曾用名 `llmuses`。如果您需要使用 `v0.4.3` 或更早版本，请运行 `pip install llmuses<=0.4.3` 并使用 `from llmuses import ...` 导入。


## 🚀 快速开始

您可以通过**命令行**或 **Python 代码**两种方式启动评测任务。

### 方式1. 使用命令行

在任意路径下执行 `evalscope eval` 命令即可开始评测。以下命令将在 `gsm8k` 和 `arc` 数据集上评测 `Qwen/Qwen2.5-0.5B-Instruct` 模型，每个数据集只取 5 个样本。

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### 方式2. 使用Python代码

使用 `run_task` 函数和 `TaskConfig` 对象来配置和启动评测任务。

```python
from evalscope import run_task TaskConfig

# 配置评测任务
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct'
    datasets=['gsm8k' 'arc']
    limit=5
)

# 启动评测
run_task(task_cfg)
```

<details><summary><b>💡 提示：</b> `run_task` 还支持字典、YAML 或 JSON 文件作为配置。</summary>

**使用 Python 字典**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct'
    'datasets': ['gsm8k' 'arc']
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**使用 YAML 文件** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### 输出结果
评测完成后，您将在终端看到如下格式的报告：
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 📈 进阶用法

### 自定义评测参数

您可以通过命令行参数精细化控制模型加载、推理和数据集配置。

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master" "precision": "torch.float16" "device_map": "auto"}' \
 --generation-config '{"do_sample":true"temperature":0.6"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0 "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: 模型加载参数，如 `revision` `precision` 等。
- `--generation-config`: 模型生成参数，如 `temperature` `max_tokens` 等。
- `--dataset-args`: 数据集配置参数，如 `few_shot_num` 等。

详情请参考 [📖 全部参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。

### 评测在线模型 API

EvalScope 支持评测通过 API 部署的模型服务（如 vLLM 部署的服务）。只需指定服务地址和 API Key 即可。

1.  **启动模型服务** (以 vLLM 为例)
    ```shell
    export VLLM_USE_MODELSCOPE=True
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen2.5 \
      --port 8801
    ```

2.  **运行评测**
    ```shell
    evalscope eval \
     --model qwen2.5 \
     --eval-type openai_api \
     --api-url http://127.0.0.1:8801/v1 \
     --api-key EMPTY \
     --datasets gsm8k \
     --limit 10
    ```

### ⚔️ 竞技场模式 (Arena)

竞技场模式通过模型间的两两对战（Pairwise Battle）来评估模型性能，并给出胜率和排名，非常适合多模型横向对比。

```text
# 评测结果示例
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
详情请参考 [📖 竞技场模式使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。

### 🖊️ 自定义数据集评测

EvalScope 允许您轻松添加和评测自己的数据集。详情请参考 [📖 自定义数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/index.html)。


## 🧪 其他评测后端
EvalScope 支持通过第三方评测框架（我们称之为“后端”）发起评测任务，以满足多样化的评测需求。

- **Native**: EvalScope 的默认评测框架，功能全面。
- **OpenCompass**: 专注于纯文本评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: 专注于多模态评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: 专注于 RAG 评测，支持 Embedding 和 Reranker 模型。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/index.html)
- **第三方评测工具**: 支持 [ToolBench](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html) 等评测任务。

## ⚡ 推理性能评测工具
EvalScope 提供了一个强大的压力测试工具，用于评估大语言模型服务的性能。

- **关键指标**: 支持吞吐量 (Tokens/s)、首字延迟 (TTFT)、Token 生成延迟 (TPOT) 等。
- **结果记录**: 支持将结果记录到 `wandb` 和 `swanlab`。
- **速度基准**: 可生成类似官方报告的速度基准测试结果。

详情请参考 [📖 性能测试使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)。

输出示例如下：
<p align="center">
    <img src="docs/zh/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>


## 📊 可视化评测结果

EvalScope 提供了一个基于 Gradio 的 WebUI，用于交互式地分析和比较评测结果。

1.  **安装依赖**
    ```bash
    pip install 'evalscope[app]'
    ```

2.  **启动服务**
    ```bash
    evalscope app
    ```
    访问 `http://127.0.0.1:7861` 即可打开可视化界面。

<table>
  <tr>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/setting.png" alt="Setting" style="width: 90%;" />
      <p>设置界面</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/model_compare.png" alt="Model Compare" style="width: 100%;" />
      <p>模型比较</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/report_overview.png" alt="Report Overview" style="width: 100%;" />
      <p>报告概览</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/report_details.png" alt="Report Details" style="width: 91%;" />
      <p>报告详情</p>
    </td>
  </tr>
</table>

详情请参考 [📖 可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)。

## 👷‍♂️ 贡献

我们欢迎来自社区的任何贡献！如果您希望添加新的评测基准、模型或功能，请参考我们的 [贡献指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)。

感谢所有为 EvalScope 做出贡献的开发者！

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>


## 📚 引用

如果您在研究中使用了 EvalScope，请引用我们的工作：
```bibtex
@misc{evalscope_2024
    title={{EvalScope}: Evaluation Framework for Large Models}
    author={ModelScope Team}
    year={2024}
    url={https://github.com/modelscope/evalscope}
}
```


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)

# Arena Mode

Arena mode allows you to configure multiple candidate models and specify a baseline model. The evaluation is conducted through pairwise battles between each candidate model and the baseline model with the win rate and ranking of each model outputted at the end. This approach is suitable for comparative evaluation among multiple models and intuitively reflects the strengths and weaknesses of each model.

## Data Preparation

To support arena mode **all candidate models need to run inference on the same dataset**. The dataset can be a general QA dataset or a domain-specific one. Below is an example using a custom `general_qa` dataset. See the [documentation](../advanced_guides/custom_dataset/llm.md#question-answering-format-qa) for details on using this dataset.

The JSONL file for the `general_qa` dataset should be in the following format. Only the `query` field is required; no additional fields are necessary. Below are two example files:

- Example content of the `arena.jsonl` file:
    ```json
    {"query": "How can I improve my time management skills?"}
    {"query": "What are the most effective ways to deal with stress?"}
    {"query": "What are the main differences between Python and JavaScript programming languages?"}
    {"query": "How can I increase my productivity while working from home?"}
    {"query": "Can you explain the basics of quantum computing?"}
    ```

- Example content of the `example.jsonl` file (with reference answers):
    ```json
    {"query": "What is the capital of France?" "response": "The capital of France is Paris."}
    {"query": "What is the largest mammal in the world?" "response": "The largest mammal in the world is the blue whale."}
    {"query": "How does photosynthesis work?" "response": "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."}
    {"query": "What is the theory of relativity?" "response": "The theory of relativity developed by Albert Einstein describes the laws of physics in relation to observers in different frames of reference."}
    {"query": "Who wrote 'To Kill a Mockingbird'?" "response": "Harper Lee wrote 'To Kill a Mockingbird'."}
    ```

## Candidate Model Inference

After preparing the dataset you can use EvalScope's `run_task` method to perform inference with the candidate models and obtain their outputs for subsequent battles.

Below is an example of how to configure inference tasks for three candidate models: `Qwen2.5-0.5B-Instruct` `Qwen2.5-7B-Instruct` and `Qwen2.5-72B-Instruct` using the same configuration for inference.

Run the following code:
```python
import os
from evalscope import TaskConfig run_task
from evalscope.constants import EvalType

models = ['qwen2.5-72b-instruct' 'qwen2.5-7b-instruct' 'qwen2.5-0.5b-instruct']

task_list = [TaskConfig(
    model=model
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=os.getenv('DASHSCOPE_API_KEY')
    eval_type=EvalType.SERVICE
    datasets=[
        'general_qa'
    ]
    dataset_args={
        'general_qa': {
            'dataset_id': 'custom_eval/text/qa'
            'subset_list': [
                'arena'
                'example'
            ]
        }
    }
    eval_batch_size=10
    generation_config={
        'temperature': 0
        'n': 1
        'max_tokens': 4096
    }) for model in models]

run_task(task_cfg=task_list)
```

<details><summary>Click to view inference results</summary>

Since the `arena` subset does not have reference answers no evaluation metrics are available for this subset. The `example` subset has reference answers so evaluation metrics will be output.
```text
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric          | Subset   |   Num |   Score | Cat.0   |
+=======================+============+=================+==========+=======+=========+=========+
| qwen2.5-0.5b-instruct | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-R       | example  |    12 |  0.8611 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-P       | example  |    12 |  0.1341 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-F       | example  |    12 |  0.1983 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-R       | example  |    12 |  0.55   | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-P       | example  |    12 |  0.0404 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-F       | example  |    12 |  0.0716 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-R       | example  |    12 |  0.8611 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-P       | example  |    12 |  0.1193 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-F       | example  |    12 |  0.1754 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-1          | example  |    12 |  0.1192 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-2          | example  |    12 |  0.0403 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-3          | example  |    12 |  0.0135 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-4          | example  |    12 |  0.0079 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-P       | example  |    12 |  0.1149 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-F       | example  |    12 |  0.1612 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-R       | example  |    12 |  0.6833 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-P       | example  |    12 |  0.0813 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-F       | example  |    12 |  0.1027 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-P       | example  |    12 |  0.101  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-F       | example  |    12 |  0.1361 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-1          | example  |    12 |  0.1009 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-2          | example  |    12 |  0.0807 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-3          | example  |    12 |  0.0625 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-4          | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-P       | example  |    12 |  0.104  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-F       | example  |    12 |  0.1418 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-R       | example  |    12 |  0.7    | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-P       | example  |    12 |  0.078  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-F       | example  |    12 |  0.0964 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-P       | example  |    12 |  0.0942 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-F       | example  |    12 |  0.1235 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-1          | example  |    12 |  0.0939 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-2          | example  |    12 |  0.0777 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-3          | example  |    12 |  0.0625 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-4          | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
```
</details>

## Candidate Model Battles

Next you can use EvalScope's `general_arena` method to conduct battles among candidate models and get their win rates and rankings on each subset. To achieve robust automatic battles you need to configure an LLM as the judge that compares the outputs of models.

During evaluation EvalScope will automatically parse the public evaluation set of candidate models use the judge model to compare the output of each candidate model with the baseline and determine which is better (to avoid model bias outputs are swapped for two rounds per comparison). The judge model's outputs are parsed as win draw or loss and each candidate model's **Elo score** and **win rate** are calculated.

Run the following code:
```python
import os
from evalscope import TaskConfig run_task

task_cfg = TaskConfig(
    model_id='Arena'  # Model ID is 'Arena'; you can omit specifying model ID
    datasets=[
        'general_arena'  # Must be 'general_arena' indicating arena mode
    ]
    dataset_args={
        'general_arena': {
            # 'system_prompt': 'xxx' # Optional: customize the judge model's system prompt here
            # 'prompt_template': 'xxx' # Optional: customize the judge model's prompt template here
            'extra_params':{
                # Configure candidate model names and corresponding report paths
                # Report paths refer to the output paths from the previous step for parsing model inference results
                'models':[
                    {
                        'name': 'qwen2.5-0.5b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-0.5b-instruct'
                    }
                    {
                        'name': 'qwen2.5-7b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-7b-instruct'
                    }
                    {
                        'name': 'qwen2.5-72b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-72b-instruct'
                    }
                ]
                # Set baseline model must be one of the candidate models
                'baseline': 'qwen2.5-7b'
            }
        }
    }
    # Configure judge model parameters
    judge_model_args={
        'model_id': 'qwen-plus'
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        'api_key': os.getenv('DASHSCOPE_API_KEY')
        'generation_config': {
            'temperature': 0.0
            'max_tokens': 8000
        }
    }
    judge_worker_num=5
    # use_cache='outputs/xxx' # Optional: to add new candidate models to existing results specify the existing results path
)

run_task(task_cfg=task_cfg)
```

<details><summary>Click to view evaluation results</summary>

```text
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Model   | Dataset       | Metric        | Subset                                     |   Num |   Score | Cat.0   |
+=========+===============+===============+============================================+=======+=========+=========+
| Arena   | general_arena | winrate       | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0185 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.5469 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.075  | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.8382 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | OVERALL                                    |    44 |  0.3617 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0185 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.3906 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.025  | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.7276 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | OVERALL                                    |    44 |  0.2826 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0909 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.6875 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.0909 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.9412 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | OVERALL                                    |    44 |  0.4469 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+ 
```
</details>


The automatically generated model leaderboard is as follows (output file located in `outputs/xxx/reports/Arena/leaderboard.txt`):

The leaderboard is sorted by win rate in descending order. As shown the `qwen2.5-72b` model performs best across all subsets with the highest win rate while the `qwen2.5-0.5b` model performs the worst.

```text
=== OVERALL LEADERBOARD ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)

=== DATASET LEADERBOARD: general_qa ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)

=== SUBSET LEADERBOARD: general_qa - example ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            54.7  (-15.6 / +14.1)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            1.8  (+0.0 / +7.2)

=== SUBSET LEADERBOARD: general_qa - arena ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            83.8  (-11.1 / +10.3)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            7.5  (-5.0 / +1.6)
```

## Visualization of Battle Results

To intuitively display the results of the battles between candidate models and the baseline EvalScope provides a visualization feature allowing you to compare the results of each candidate model against the baseline model for each sample.

Run the command below to launch the visualization interface:
```shell
evalscope app
```
Open `http://localhost:7860` in your browser to view the visualization page.

Workflow:
1. Select the latest `general_arena` evaluation report and click the "Load and View" button.
2. Click dataset details and select the battle results between your candidate model and the baseline.
3. Adjust the threshold to filter battle results (normalized scores range from 0-1; 0.5 indicates a tie scores above 0.5 indicate the candidate is better than the baseline below 0.5 means worse).

Example below: a battle between `qwen2.5-72b` and `qwen2.5-7b`. The model judged the 72b as better:

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/arena_example.jpg)


# Sandbox Environment Usage

To complete LLM code capability evaluation we need to set up an independent evaluation environment to avoid executing erroneous code in the development environment and causing unavoidable losses. Currently EvalScope has integrated the [ms-enclave](https://github.com/modelscope/ms-enclave) sandbox environment allowing users to evaluate model code capabilities in a controlled environment such as using evaluation benchmarks like HumanEval and LiveCodeBench.

The following introduces two different sandbox usage methods:

- Local usage: Set up the sandbox environment on a local machine and conduct evaluation locally requiring Docker support on the local machine;
- Remote usage: Set up the sandbox environment on a remote server and conduct evaluation through API interfaces requiring Docker support on the remote machine.

## 1. Local Usage

Use Docker to set up a sandbox environment on a local machine and conduct evaluation locally requiring Docker support on the local machine.

### Environment Setup

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in your local Python environment:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration
When running evaluations add the `use_sandbox` and `sandbox_type` parameters to automatically enable the sandbox environment. Other parameters remain the same as regular evaluations:

Here's a complete example code for model evaluation on HumanEval:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig run_task

task_config = TaskConfig(
    model='qwen-plus'
    datasets=['humaneval']
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=env.get('DASHSCOPE_API_KEY')
    eval_type='openai_api'
    eval_batch_size=5
    limit=5
    generation_config={
        'max_tokens': 4096
        'temperature': 0.0
        'seed': 42
    }
    use_sandbox=True # enable sandbox
    sandbox_type='docker' # specify sandbox type
    judge_worker_num=5 # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation EvalScope will automatically start and manage the sandbox environment ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] Local sandbox manager started
...
```

## 2. Remote Usage

Set up the sandbox environment on a remote server and conduct evaluation through API interfaces requiring Docker support on the remote machine.

### Environment Setup

You need to install and configure separately on both the remote machine and local machine.

#### Remote Machine

The environment installation on the remote machine is similar to the local usage method described above:

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in remote Python environment:

```bash
pip install evalscope[sandbox]
```

3. **Start sandbox server**: Run the following command to start the sandbox server:

```bash
ms-enclave server --host 0.0.0.0 --port 1234
```

#### Local Machine

The local machine does not need Docker installation at this point but needs to install EvalScope:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration

When running evaluations add the `use_sandbox` parameter to automatically enable the sandbox environment and specify the remote sandbox server's API address in `sandbox_manager_config`:

Complete example code is as follows:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig run_task

task_config = TaskConfig(
    model='qwen-plus'
    datasets=['humaneval']
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=env.get('DASHSCOPE_API_KEY')
    eval_type='openai_api'
    eval_batch_size=5
    limit=5
    generation_config={
        'max_tokens': 4096
        'temperature': 0.0
        'seed': 42
    }
    use_sandbox=True # enable sandbox
    sandbox_type='docker' # specify sandbox type
    sandbox_manager_config={
        'base_url': 'http://<remote_host>:1234'  # remote sandbox manager URL
    }
    judge_worker_num=5 # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation EvalScope will communicate with the remote sandbox server through API ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] HTTP sandbox manager started connected to http://<remote_host>:1234
...
```


# EvalScope Service Deployment

## Introduction

EvalScope service mode provides HTTP API-based evaluation and stress testing capabilities designed to address the following scenarios:

1. **Remote Invocation**: Support remote evaluation functionality through network without configuring complex evaluation environments locally
2. **Service Integration**: Easily integrate evaluation capabilities into existing workflows CI/CD pipelines or automated testing systems
3. **Multi-user Collaboration**: Support multiple users or systems calling the evaluation service simultaneously improving resource utilization
4. **Unified Management**: Centrally manage evaluation resources and configurations for easier maintenance and monitoring
5. **Flexible Deployment**: Can be deployed on dedicated servers or container environments decoupled from business systems

The Flask service encapsulates EvalScope's core evaluation (eval) and stress testing (perf) functionalities providing services through standard RESTful APIs making evaluation capabilities callable and integrable like other microservices.

## Features

- **Model Evaluation** (`/api/v1/eval`): Support evaluation of OpenAI API-compatible models
- **Performance Testing** (`/api/v1/perf`): Support performance benchmarking of OpenAI API-compatible models
- **Parameter Query**: Provide parameter description endpoints

## Environment Setup


### Full Installation (Recommended)

```bash
pip install evalscope[service]
```

### Development Environment Installation

```bash
# Clone repository
git clone https://github.com/modelscope/evalscope.git
cd evalscope

# Install development version with service
pip install -e '.[service]'
```

## Starting the Service

### Command Line Launch

```bash
# Use default configuration (host: 0.0.0.0 port: 9000)
evalscope service

# Custom host and port
evalscope service --host 127.0.0.1 --port 9000

# Enable debug mode
evalscope service --debug
```

### Python Code Launch

```python
from evalscope.service import run_service

# Start service
run_service(host='0.0.0.0' port=9000 debug=False)
```

## API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response Example:**
```json
{
  "status": "ok"
  "service": "evalscope"
  "timestamp": "2025-12-04T10:00:00"
}
```

### 2. Model Evaluation

```bash
POST /api/v1/eval
```

**Request Body Example:**
```json
{
  "model": "qwen-plus"
  "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
  "api_key": "your-api-key"
  "datasets": ["gsm8k" "iquiz"]
  "limit": 10
  "generation_config": {
    "temperature": 0.0
    "max_tokens": 2048
  }
}
```

**Required Parameters:**
- `model`: Model name
- `datasets`: List of datasets
- `api_url`: API endpoint URL (OpenAI-compatible)

**Optional Parameters:**
- `api_key`: API key (default: "EMPTY")
- `limit`: Evaluation sample quantity limit
- `eval_batch_size`: Batch size (default: 1)
- `generation_config`: Generation configuration
  - `temperature`: Temperature parameter (default: 0.0)
  - `max_tokens`: Maximum generation tokens (default: 2048)
  - `top_p`: Nucleus sampling parameter
  - `top_k`: Top-k sampling parameter
- `work_dir`: Output directory
- `debug`: Debug mode
- `seed`: Random seed (default: 42)

**Response Example:**
```json
{
  "status": "success"
  "message": "Evaluation completed"
  "result": {"...": "..."}
  "output_dir": "/path/to/outputs/20251204_100000"
}
```

### 3. Performance Testing

```bash
POST /api/v1/perf
```

**Request Body Example:**
```json
{
  "model": "qwen-plus"
  "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  "api": "openai"
  "api_key": "your-api-key"
  "number": 100
  "parallel": 10
  "dataset": "openqa"
  "max_tokens": 2048
  "temperature": 0.0
}
```

**Required Parameters:**
- `model`: Model name
- `url`: Complete API endpoint URL

**Optional Parameters:**
- `api`: API type (openai/dashscope/anthropic/gemini default: "openai")
- `api_key`: API key
- `number`: Total number of requests (default: 1000)
- `parallel`: Concurrency level (default: 1)
- `rate`: Requests per second limit (default: -1 unlimited)
- `dataset`: Dataset name (default: "openqa")
- `max_tokens`: Maximum generation tokens (default: 2048)
- `temperature`: Temperature parameter (default: 0.0)
- `stream`: Whether to use streaming output (default: true)
- `debug`: Debug mode

**Response Example:**
```json
{
  "status": "success"
  "message": "Performance test completed"
  "output_dir": "/path/to/outputs"
  "results": {
    "parallel_10_number_100": {
      "metrics": {"...": "..."}
      "percentiles": {"...": "..."}
    }
  }
}
```

### 4. Get Evaluation Parameter Description

```bash
GET /api/v1/eval/params
```

Returns descriptions of all parameters supported by the evaluation endpoint.

### 5. Get Performance Test Parameter Description

```bash
GET /api/v1/perf/params
```

Returns descriptions of all parameters supported by the performance test endpoint.

## Usage Examples

### Testing Evaluation Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "api_key": "your-api-key"
    "datasets": ["gsm8k"]
    "limit": 5
  }'
```

### Testing Performance Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/perf \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    "api": "openai"
    "number": 50
    "parallel": 5
  }'
```

### Using Python requests

```python
import requests

# Evaluation request
eval_response = requests.post(
    'http://localhost:9000/api/v1/eval'
    json={
        'model': 'qwen-plus'
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        'api_key': 'your-api-key'
        'datasets': ['gsm8k' 'iquiz']
        'limit': 10
        'generation_config': {
            'temperature': 0.0
            'max_tokens': 2048
        }
    }
)
print(eval_response.json())

# Performance test request
perf_response = requests.post(
    'http://localhost:9000/api/v1/perf'
    json={
        'model': 'qwen-plus'
        'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        'api': 'openai'
        'number': 100
        'parallel': 10
        'dataset': 'openqa'
    }
)
print(perf_response.json())
```

## Important Notes

1. **OpenAI API-Compatible Models Only**: This service is designed specifically for OpenAI API-compatible models
2. **Long-Running Tasks**: Evaluation and performance testing tasks may take considerable time. We recommend setting appropriate HTTP timeout values on the client side as the API calls are synchronous and will block until completion.
3. **Output Directory**: Evaluation results are saved in the configured `work_dir` default is `outputs/`
4. **Error Handling**: The service returns detailed error messages and stack traces (in debug mode)
5. **Resource Management**: Pay attention to concurrency settings during stress testing to avoid server overload

## Error Codes

- `400`: Invalid request parameters
- `404`: Endpoint not found
- `500`: Internal server error

## Example Scenarios

### Scenario 1: Quick Evaluation of Qwen Model

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "api_key": "sk-..."
    "datasets": ["gsm8k"]
    "limit": 100
  }'
```

### Scenario 2: Stress Testing Locally Deployed Model

```bash
curl -X POST http://localhost:9000/api/v1/perf \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5"
    "url": "http://localhost:8000/v1/chat/completions"
    "api": "openai"
    "number": 1000
    "parallel": 20
    "max_tokens": 2048
  }'
```

### Scenario 3: Multi-Dataset Evaluation

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "datasets": ["gsm8k" "iquiz" "ceval"]
    "limit": 50
    "eval_batch_size": 4
  }'
```

<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  Chinese Documentation</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documentation</a>
<p>


> ⭐ If you like this project please click the "Star" button in the upper right corner to support us. Your support is our motivation to move forward!

## 📝 Introduction

EvalScope is a powerful and easily extensible model evaluation framework created by the [ModelScope Community](https://modelscope.cn/) aiming to provide a one-stop evaluation solution for large model developers.

Whether you want to evaluate the general capabilities of models conduct multi-model performance comparisons or need to stress test models EvalScope can meet your needs.

## ✨ Key Features

- **📚 Comprehensive Evaluation Benchmarks**: Built-in multiple industry-recognized evaluation benchmarks including MMLU C-Eval GSM8K and more.
- **🧩 Multi-modal and Multi-domain Support**: Supports evaluation of various model types including Large Language Models (LLM) Vision Language Models (VLM) Embedding Reranker AIGC and more.
- **🚀 Multi-backend Integration**: Seamlessly integrates multiple evaluation backends including OpenCompass VLMEvalKit RAGEval to meet different evaluation needs.
- **⚡ Inference Performance Testing**: Provides powerful model service stress testing tools supporting multiple performance metrics such as TTFT TPOT.
- **📊 Interactive Reports**: Provides WebUI visualization interface supporting multi-dimensional model comparison report overview and detailed inspection.
- **⚔️ Arena Mode**: Supports multi-model battles (Pairwise Battle) intuitively ranking and evaluating models.
- **🔧 Highly Extensible**: Developers can easily add custom datasets models and evaluation metrics.

<details><summary>🏛️ Overall Architecture</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope Overall Architecture.
</p>

1.  **Input Layer**
    - **Model Sources**: API models (OpenAI API) Local models (ModelScope)
    - **Datasets**: Standard evaluation benchmarks (MMLU/GSM8k etc.) Custom data (MCQ/QA)

2.  **Core Functions**
    - **Multi-backend Evaluation**: Native backend OpenCompass MTEB VLMEvalKit RAGAS
    - **Performance Monitoring**: Supports multiple model service APIs and data formats tracking TTFT/TPOP and other metrics
    - **Tool Extensions**: Integrates Tool-Bench Needle-in-a-Haystack etc.

3.  **Output Layer**
    - **Structured Reports**: Supports JSON Table Logs
    - **Visualization Platform**: Supports Gradio Wandb SwanLab

</details>

## 🎉 What's New

> [!IMPORTANT]
> **Version 1.0 Refactoring**
>
> Version 1.0 introduces a major overhaul of the evaluation framework establishing a new more modular and extensible API layer under `evalscope/api`. Key improvements include standardized data models for benchmarks samples and results; a registry-based design for components such as benchmarks and metrics; and a rewritten core evaluator that orchestrates the new architecture. Existing benchmark adapters have been migrated to this API resulting in cleaner more consistent and easier-to-maintain implementations.

- 🔥 **[2025.12.02]** Added support for custom multimodal VQA evaluation; refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html). Added support for visualizing model service stress testing in ClearML; refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#clearml).
- 🔥 **[2025.11.26]** Added support for OpenAI-MRCR GSM8K-V MGSM MicroVQA IFBench SciCode benchmarks.
- 🔥 **[2025.11.18]** Added support for custom Function-Call (tool invocation) datasets to test whether models can timely and correctly call tools. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#function-calling-format-fc).
- 🔥 **[2025.11.14]** Added support for SWE-bench_Verified SWE-bench_Lite SWE-bench_Verified_mini code evaluation benchmarks. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html).
- 🔥 **[2025.11.12]** Added `pass@k` `vote@k` `pass^k` and other metric aggregation methods; added support for multimodal evaluation benchmarks such as A_OKVQA CMMU ScienceQA V*Bench.
- 🔥 **[2025.11.07]** Added support for τ²-bench an extended and enhanced version of τ-bench that includes a series of code fixes and adds telecom domain troubleshooting scenarios. Refer to the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/tau2_bench.html).
- 🔥 **[2025.10.30]** Added support for BFCL-v4 enabling evaluation of agent capabilities including web search and long-term memory. See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v4.html).
- 🔥 **[2025.10.27]** Added support for LogiQA HaluEval MathQA MRI-QA PIQA QASC CommonsenseQA and other evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.26]** Added support for Conll-2003 CrossNER Copious GeniaNER HarveyNER MIT-Movie-Trivia MIT-Restaurant OntoNotes5 WNUT2017 and other Named Entity Recognition evaluation benchmarks. Thanks to @[penguinwang96825](https://github.com/penguinwang96825) for the code implementation.
- 🔥 **[2025.10.21]** Optimized sandbox environment usage in code evaluation supporting both local and remote operation modes. For details refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html).
- 🔥 **[2025.10.20]** Added support for evaluation benchmarks including PolyMath SimpleVQA MathVerse MathVision AA-LCR; optimized evalscope perf performance to align with vLLM Bench. For details refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/vs_vllm_bench.html).
- 🔥 **[2025.10.14]** Added support for OCRBench OCRBench-v2 DocVQA InfoVQA ChartQA and BLINK multimodal image-text evaluation benchmarks.
- 🔥 **[2025.09.22]** Code evaluation benchmarks (HumanEval LiveCodeBench) now support running in a sandbox environment. To use this feature please install [ms-enclave](https://github.com/modelscope/ms-enclave) first.
- 🔥 **[2025.09.19]** Added support for multimodal image-text evaluation benchmarks including RealWorldQA AI2D MMStar MMBench and OmniBench as well as pure text evaluation benchmarks such as Multi-IF HealthBench and AMC.
- 🔥 **[2025.09.05]** Added support for vision-language multimodal model evaluation tasks such as MathVista and MMMU. For more supported datasets please [refer to the documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/vlm.html).
- 🔥 **[2025.09.04]** Added support for image editing task evaluation including the [GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) benchmark. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/image_edit.html).
- 🔥 **[2025.08.22]** Version 1.0 Refactoring. Break changes please [refer to](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#switching-to-version-v1-0).
<details><summary>More</summary>

- 🔥 **[2025.07.18]** The model stress testing now supports randomly generating image-text data for multimodal model evaluation. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#id4).
- 🔥 **[2025.07.16]** Support for [τ-bench](https://github.com/sierra-research/tau-bench) has been added enabling the evaluation of AI Agent performance and reliability in real-world scenarios involving dynamic user and tool interactions. For usage instructions please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html#bench).
- 🔥 **[2025.07.14]** Support for "Humanity's Last Exam" ([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle)) a highly challenging evaluation benchmark. For usage instructions refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam).
- 🔥 **[2025.07.03]** Refactored Arena Mode: now supports custom model battles outputs a model leaderboard and provides battle result visualization. See [reference](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html) for details.
- 🔥 **[2025.06.28]** Optimized custom dataset evaluation: now supports evaluation without reference answers. Enhanced LLM judge usage with built-in modes for "scoring directly without reference answers" and "checking answer consistency with reference answers". See [reference](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/llm.html#qa) for details.
- 🔥 **[2025.06.19]** Added support for the [BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3) benchmark designed to evaluate model function-calling capabilities across various scenarios. For more information refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html).
- 🔥 **[2025.06.02]** Added support for the Needle-in-a-Haystack test. Simply specify `needle_haystack` to conduct the test and a corresponding heatmap will be generated in the `outputs/reports` folder providing a visual representation of the model's performance. Refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/needle_haystack.html) for more details.
- 🔥 **[2025.05.29]** Added support for two long document evaluation benchmarks: [DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary) and [FRAMES](https://modelscope.cn/datasets/iic/frames/summary). For usage guidelines please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html).
- 🔥 **[2025.05.16]** Model service performance stress testing now supports setting various levels of concurrency and outputs a performance test report. [Reference example](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html#id3).
- 🔥 **[2025.05.13]** Added support for the [ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static) dataset to evaluate model's tool-calling capabilities. Refer to the [documentation](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html) for usage instructions. Also added support for the [DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview) and [Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val) benchmarks to assess the reasoning capabilities of models.
- 🔥 **[2025.04.29]** Added Qwen3 Evaluation Best Practices [welcome to read 📖](https://evalscope.readthedocs.io/en/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** Support for text-to-image evaluation: Supports 8 metrics including MPS HPSv2.1Score etc. and evaluation benchmarks such as EvalMuse GenAI-Bench. Refer to the [user documentation](https://evalscope.readthedocs.io/en/latest/user_guides/aigc/t2i.html) for more details.
- 🔥 **[2025.04.10]** Model service stress testing tool now supports the `/v1/completions` endpoint (the default endpoint for vLLM benchmarking)
- 🔥 **[2025.04.08]** Support for evaluating embedding model services compatible with the OpenAI API has been added. For more details check the [user guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters).
- 🔥 **[2025.03.27]** Added support for [AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview) and [ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary) evaluation benchmarks. For usage notes please refer to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** The model inference service stress testing now supports generating prompts of specified length using random values. Refer to the [user guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html#using-the-random-dataset) for more details.
- 🔥 **[2025.03.13]** Added support for the [LiveCodeBench](https://www.modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary) code evaluation benchmark which can be used by specifying `live_code_bench`. Supports evaluating QwQ-32B on LiveCodeBench refer to the [best practices](https://evalscope.readthedocs.io/en/latest/best_practice/eval_qwq.html).
- 🔥 **[2025.03.11]** Added support for the [SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary) and [Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary) evaluation benchmarks. These are used to assess the factual accuracy of models and you can specify `simple_qa` and `chinese_simpleqa` for use. Support for specifying a judge model is also available. For more details refer to the [relevant parameter documentation](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).
- 🔥 **[2025.03.07]** Added support for the [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/summary) model evaluate the model's reasoning ability and reasoning efficiency refer to [📖 Best Practices for QwQ-32B Evaluation](https://evalscope.readthedocs.io/en/latest/best_practice/eval_qwq.html) for more details.
- 🔥 **[2025.03.04]** Added support for the [SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary) dataset which covers 13 categories 72 first-level disciplines and 285 second-level disciplines totaling 26529 questions. You can use it by specifying `super_gpqa`.
- 🔥 **[2025.03.03]** Added support for evaluating the IQ and EQ of models. Refer to [📖 Best Practices for IQ and EQ Evaluation](https://evalscope.readthedocs.io/en/latest/best_practice/iquiz.html) to find out how smart your AI is!
- 🔥 **[2025.02.27]** Added support for evaluating the reasoning efficiency of models. Refer to [📖 Best Practices for Evaluating Thinking Efficiency](https://evalscope.readthedocs.io/en/latest/best_practice/think_eval.html). This implementation is inspired by the works [Overthinking](https://doi.org/10.48550/arXiv.2412.21187) and [Underthinking](https://doi.org/10.48550/arXiv.2501.18585).
- 🔥 **[2025.02.25]** Added support for two model inference-related evaluation benchmarks: [MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR) and [ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary). To use them simply specify `musr` and `process_bench` respectively in the datasets parameter.
- 🔥 **[2025.02.18]** Supports the AIME25 dataset which contains 15 questions (Grok3 scored 93 on this dataset).
- 🔥 **[2025.02.13]** Added support for evaluating DeepSeek distilled models including AIME24 MATH-500 and GPQA-Diamond datasets，refer to [best practice](https://evalscope.readthedocs.io/en/latest/best_practice/deepseek_r1_distill.html); Added support for specifying the `eval_batch_size` parameter to accelerate model evaluation.
- 🔥 **[2025.01.20]** Support for visualizing evaluation results including single model evaluation results and multi-model comparison refer to the [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html) for more details; Added [`iquiz`](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary) evaluation example evaluating the IQ and EQ of the model.
- 🔥 **[2025.01.07]** Native backend: Support for model API evaluation is now available. Refer to the [📖 Model API Evaluation Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#api) for more details. Additionally support for the `ifeval` evaluation benchmark has been added.
- 🔥🔥 **[2024.12.31]** Support for adding benchmark evaluations refer to the [📖 Benchmark Evaluation Addition Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html); support for custom mixed dataset evaluations allowing for more comprehensive model evaluations with less data refer to the [📖 Mixed Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/collection/index.html).
- 🔥 **[2024.12.13]** Model evaluation optimization: no need to pass the `--template-type` parameter anymore; supports starting evaluation with `evalscope eval --args`. Refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html) for more details.
- 🔥 **[2024.11.26]** The model inference service performance evaluator has been completely refactored: it now supports local inference service startup and Speed Benchmark; asynchronous call error handling has been optimized. For more details refer to the [📖 User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).
- 🔥 **[2024.10.31]** The best practice for evaluating Multimodal-RAG has been updated please check the [📖 Blog](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag) for more details.
- 🔥 **[2024.10.23]** Supports multimodal RAG evaluation including the assessment of image-text retrieval using [CLIP_Benchmark](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/clip_benchmark.html) and extends [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html) to support end-to-end multimodal metrics evaluation.
- 🔥 **[2024.10.8]** Support for RAG evaluation including independent evaluation of embedding models and rerankers using [MTEB/CMTEB](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/mteb.html) as well as end-to-end evaluation using [RAGAS](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/ragas.html).
- 🔥 **[2024.09.18]** Our documentation has been updated to include a blog module featuring some technical research and discussions related to evaluations. We invite you to [📖 read it](https://evalscope.readthedocs.io/en/refact_readme/blog/index.html).
- 🔥 **[2024.09.12]** Support for LongWriter evaluation which supports 10000+ word generation. You can use the benchmark [LongBench-Write](evalscope/third_party/longbench_write/README.md) to measure the long output quality as well as the output length.
- 🔥 **[2024.08.30]** Support for custom dataset evaluations including text datasets and multimodal image-text datasets.
- 🔥 **[2024.08.20]** Updated the official documentation including getting started guides best practices and FAQs. Feel free to [📖read it here](https://evalscope.readthedocs.io/en/latest/)!
- 🔥 **[2024.08.09]** Simplified the installation process allowing for pypi installation of vlmeval dependencies; optimized the multimodal model evaluation experience achieving up to 10x acceleration based on the OpenAI API evaluation chain.
- 🔥 **[2024.07.31]** Important change: The package name `llmuses` has been changed to `evalscope`. Please update your code accordingly.
- 🔥 **[2024.07.26]** Support for **VLMEvalKit** as a third-party evaluation framework to initiate multimodal model evaluation tasks.
- 🔥 **[2024.06.29]** Support for **OpenCompass** as a third-party evaluation framework which we have encapsulated at a higher level supporting pip installation and simplifying evaluation task configuration.
- 🔥 **[2024.06.13]** EvalScope seamlessly integrates with the fine-tuning framework SWIFT providing full-chain support from LLM training to evaluation.
- 🔥 **[2024.06.13]** Integrated the Agent evaluation dataset ToolBench.

</details>

## ❤️ Community & Support

Welcome to join our community to communicate with other developers and get help.

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  WeChat Group | DingTalk Group
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">



## 🛠️ Environment Setup

We recommend using `conda` to create a virtual environment and install with `pip`.

1.  **Create and Activate Conda Environment** (Python 3.10 recommended)
    ```shell
    conda create -n evalscope python=3.10
    conda activate evalscope
    ```

2.  **Install EvalScope**

    - **Method 1: Install via PyPI (Recommended)**
      ```shell
      pip install evalscope
      ```

    - **Method 2: Install from Source (For Development)**
      ```shell
      git clone https://github.com/modelscope/evalscope.git
      cd evalscope
      pip install -e .
      ```

3.  **Install Additional Dependencies** (Optional)
    Install corresponding feature extensions according to your needs:
    ```shell
    # Performance testing
    pip install 'evalscope[perf]'

    # Visualization App
    pip install 'evalscope[app]'

    # Other evaluation backends
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'

    # Install all dependencies
    pip install 'evalscope[all]'
    ```
    > If you installed from source please replace `evalscope` with `.` for example `pip install '.[perf]'`.

> [!NOTE]
> This project was formerly known as `llmuses`. If you need to use `v0.4.3` or earlier versions please run `pip install llmuses<=0.4.3` and use `from llmuses import ...` for imports.


## 🚀 Quick Start

You can start evaluation tasks in two ways: **command line** or **Python code**.

### Method 1. Using Command Line

Execute the `evalscope eval` command in any path to start evaluation. The following command will evaluate the `Qwen/Qwen2.5-0.5B-Instruct` model on `gsm8k` and `arc` datasets taking only 5 samples from each dataset.

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### Method 2. Using Python Code

Use the `run_task` function and `TaskConfig` object to configure and start evaluation tasks.

```python
from evalscope import run_task TaskConfig

# Configure evaluation task
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct'
    datasets=['gsm8k' 'arc']
    limit=5
)

# Start evaluation
run_task(task_cfg)
```

<details><summary><b>💡 Tip:</b> `run_task` also supports dictionaries YAML or JSON files as configuration.</summary>

**Using Python Dictionary**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct'
    'datasets': ['gsm8k' 'arc']
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**Using YAML File** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### Output Results
After evaluation completion you will see a report in the terminal in the following format:
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 📈 Advanced Usage

### Custom Evaluation Parameters

You can fine-tune model loading inference and dataset configuration through command line parameters.

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master" "precision": "torch.float16" "device_map": "auto"}' \
 --generation-config '{"do_sample":true"temperature":0.6"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0 "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: Model loading parameters such as `revision` `precision` etc.
- `--generation-config`: Model generation parameters such as `temperature` `max_tokens` etc.
- `--dataset-args`: Dataset configuration parameters such as `few_shot_num` etc.

For details please refer to [📖 Complete Parameter Guide](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).

### Evaluating Online Model APIs

EvalScope supports evaluating model services deployed via APIs (such as services deployed with vLLM). Simply specify the service address and API Key.

1.  **Start Model Service** (using vLLM as example)
    ```shell
    export VLLM_USE_MODELSCOPE=True
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen2.5 \
      --port 8801
    ```

2.  **Run Evaluation**
    ```shell
    evalscope eval \
     --model qwen2.5 \
     --eval-type openai_api \
     --api-url http://127.0.0.1:8801/v1 \
     --api-key EMPTY \
     --datasets gsm8k \
     --limit 10
    ```

### ⚔️ Arena Mode

Arena mode evaluates model performance through pairwise battles between models providing win rates and rankings perfect for horizontal comparison of multiple models.

```text
# Example evaluation results
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
For details please refer to [📖 Arena Mode Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html).

### 🖊️ Custom Dataset Evaluation

EvalScope allows you to easily add and evaluate your own datasets. For details please refer to [📖 Custom Dataset Evaluation Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html).


## 🧪 Other Evaluation Backends
EvalScope supports launching evaluation tasks through third-party evaluation frameworks (we call them "backends") to meet diverse evaluation needs.

- **Native**: EvalScope's default evaluation framework with comprehensive functionality.
- **OpenCompass**: Focuses on text-only evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: Focuses on multi-modal evaluation. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: Focuses on RAG evaluation supporting Embedding and Reranker models. [📖 Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/rageval_backend/index.html)
- **Third-party Evaluation Tools**: Supports evaluation tasks like [ToolBench](https://evalscope.readthedocs.io/en/latest/third_party/toolbench.html).

## ⚡ Inference Performance Evaluation Tool
EvalScope provides a powerful stress testing tool for evaluating the performance of large language model services.

- **Key Metrics**: Supports throughput (Tokens/s) first token latency (TTFT) token generation latency (TPOT) etc.
- **Result Recording**: Supports recording results to `wandb` and `swanlab`.
- **Speed Benchmarks**: Can generate speed benchmark results similar to official reports.

For details please refer to [📖 Performance Testing Usage Guide](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).

Example output is shown below:
<p align="center">
    <img src="docs/en/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>


## 📊 Visualizing Evaluation Results

EvalScope provides a Gradio-based WebUI for interactive analysis and comparison of evaluation results.

1.  **Install Dependencies**
    ```bash
    pip install 'evalscope[app]'
    ```

2.  **Start Service**
    ```bash
    evalscope app
    ```
    Visit `http://127.0.0.1:7861` to open the visualization interface.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/setting.png" alt="Setting" style="width: 85%;" />
      <p>Settings Interface</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/model_compare.png" alt="Model Compare" style="width: 100%;" />
      <p>Model Comparison</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/report_overview.png" alt="Report Overview" style="width: 100%;" />
      <p>Report Overview</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/en/get_started/images/report_details.png" alt="Report Details" style="width: 85%;" />
      <p>Report Details</p>
    </td>
  </tr>
</table>

For details please refer to [📖 Visualizing Evaluation Results](https://evalscope.readthedocs.io/en/latest/get_started/visualization.html).

## 👷‍♂️ Contributing

We welcome any contributions from the community! If you want to add new evaluation benchmarks models or features please refer to our [Contributing Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/add_benchmark.html).

Thanks to all developers who have contributed to EvalScope!

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>


## 📚 Citation

If you use EvalScope in your research please cite our work:
```bibtex
@misc{evalscope_2024
    title={{EvalScope}: Evaluation Framework for Large Models}
    author={ModelScope Team}
    year={2024}
    url={https://github.com/modelscope/evalscope}
}
```


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)

<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  中文 &nbsp ｜ &nbsp <a href="evalscope.md">English</a> &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/zh-cn/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documents</a>
<p>


> ⭐ 如果你喜欢这个项目，请点击右上角的 "Star" 按钮支持我们。你的支持是我们前进的动力！

## 📝 简介

EvalScope 是由[魔搭社区](https://modelscope.cn/)打造的一款功能强大、易于扩展的模型评测框架，旨在为大模型开发者提供一站式评测解决方案。

无论您是想评估模型的通用能力、进行多模型性能对比，还是需要对模型进行压力测试，EvalScope 都能满足您的需求。

## ✨ 主要特性

- **📚 全面的评测基准**: 内置 MMLU C-Eval GSM8K 等多个业界公认的评测基准。
- **🧩 多模态与多领域支持**: 支持大语言模型 (LLM)、多模态 (VLM)、Embedding、Reranker、AIGC 等多种模型的评测。
- **🚀 多后端集成**: 无缝集成 OpenCompass VLMEvalKit RAGEval 等多种评测后端，满足不同评测需求。
- **⚡ 推理性能测试**: 提供强大的模型服务压力测试工具，支持 TTFT TPOT 等多项性能指标。
- **📊 交互式报告**: 提供 WebUI 可视化界面，支持多维度模型对比、报告概览和详情查阅。
- **⚔️ 竞技场模式**: 支持多模型对战 (Pairwise Battle)，直观地对模型进行排名和评估。
- **🔧 高度可扩展**: 开发者可以轻松添加自定义数据集、模型和评测指标。

<details><summary>🏛️ 整体架构</summary>

<p align="center">
    <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/EvalScope%E6%9E%B6%E6%9E%84%E5%9B%BE.png" style="width: 70%;">
    <br>EvalScope 整体架构图.
</p>

1.  **输入层**
    - **模型来源**: API模型（OpenAI API）、本地模型（ModelScope）
    - **数据集**: 标准评测基准（MMLU/GSM8k等）、自定义数据（MCQ/QA）

2.  **核心功能**
    - **多后端评估**: 原生后端、OpenCompass、MTEB、VLMEvalKit、RAGAS
    - **性能监控**: 支持多种模型服务 API 和数据格式，追踪 TTFT/TPOP 等指标
    - **工具扩展**: 集成 Tool-Bench Needle-in-a-Haystack 等

3.  **输出层**
    - **结构化报告**: 支持 JSON Table Logs
    - **可视化平台**: 支持 Gradio Wandb SwanLab

</details>

## 🎉 内容更新

> [!IMPORTANT]
> **版本 1.0 重构**
>
> 版本 1.0 对评测框架进行了重大重构，在 `evalscope/api` 下建立了全新的、更模块化且易扩展的 API 层。主要改进包括：为基准、样本和结果引入了标准化数据模型；对基准和指标等组件采用注册表式设计；并重写了核心评测器以协同新架构。现有的基准已迁移到这一 API，实现更加简洁、一致且易于维护。

- 🔥 **[2025.12.02]** 支持自定义多模态VQA评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/vlm.html) ；支持模型服务压测在 ClearML 上可视化，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#clearml)。
- 🔥 **[2025.11.26]** 新增支持 OpenAI-MRCR、GSM8K-V、MGSM、MicroVQA、IFBench、SciCode 评测基准。
- 🔥 **[2025.11.18]** 支持自定义 Function-Call（工具调用）数据集，来测试模型能否适时并正确调用工具，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#fc)
- 🔥 **[2025.11.14]** 新增支持SWE-bench_Verified SWE-bench_Lite SWE-bench_Verified_mini 代码评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)。
- 🔥 **[2025.11.12]** 新增`pass@k`、`vote@k`、`pass^k`等指标聚合方法；新增支持A_OKVQA CMMU ScienceQ V*Bench等多模态评测基准。
- 🔥 **[2025.11.07]** 新增支持τ²-bench，是 τ-bench 的扩展与增强版本，包含一系列代码修复，并新增了电信（telecom）领域的故障排查场景，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/tau2_bench.html)。
- 🔥 **[2025.10.30]** 新增支持BFCL-v4，支持agent的网络搜索和长期记忆能力的评测，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v4.html)。
- 🔥 **[2025.10.27]** 新增支持LogiQA HaluEval MathQA MRI-QA PIQA QASC CommonsenseQA等评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.26]** 新增支持Conll-2003 CrossNER Copious GeniaNER HarveyNER MIT-Movie-Trivia MIT-Restaurant OntoNotes5 WNUT2017 等命名实体识别评测基准。感谢 @[penguinwang96825](https://github.com/penguinwang96825) 提供代码实现。
- 🔥 **[2025.10.21]** 优化代码评测中的沙箱环境使用，支持在本地和远程两种模式下运行，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 🔥 **[2025.10.20]** 新增支持PolyMath SimpleVQA MathVerse MathVision AA-LCR 等评测基准；优化evalscope perf表现，对齐vLLM Bench，具体参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/vs_vllm_bench.html)。
- 🔥 **[2025.10.14]** 新增支持OCRBench OCRBench-v2 DocVQA InfoVQA ChartQA BLINK 等图文多模态评测基准。
- 🔥 **[2025.09.22]** 代码评测基准(HumanEval LiveCodeBench)支持在沙箱环境中运行，要使用该功能需先安装[ms-enclave](https://github.com/modelscope/ms-enclave)。
- 🔥 **[2025.09.19]** 新增支持RealWorldQA、AI2D、MMStar、MMBench、OmniBench等图文多模态评测基准，和Multi-IF、HealthBench、AMC等纯文本评测基准。
- 🔥 **[2025.09.05]** 支持视觉-语言多模态大模型的评测任务，例如：MathVista、MMMU，更多支持数据集请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/vlm.html)。
- 🔥 **[2025.09.04]** 支持图像编辑任务评测，支持[GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench) 评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/image_edit.html)。
- 🔥 **[2025.08.22]** Version 1.0 重构，不兼容的更新请[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#v1-0)。
<details> <summary>更多</summary>

- 🔥 **[2025.07.18]** 模型压测支持随机生成图文数据，用于多模态模型压测，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#id4)。
- 🔥 **[2025.07.16]** 支持[τ-bench](https://github.com/sierra-research/tau-bench)，用于评估 AI Agent在动态用户和工具交互的实际环境中的性能和可靠性，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#bench)。
- 🔥 **[2025.07.14]** 支持“人类最后的考试”([Humanity's-Last-Exam](https://modelscope.cn/datasets/cais/hle))，这一高难度评测基准，使用方法[参考](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/llm.html#humanity-s-last-exam)。
- 🔥 **[2025.07.03]** 重构了竞技场模式，支持自定义模型对战，输出模型排行榜，以及对战结果可视化，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。
- 🔥 **[2025.06.28]** 优化自定义数据集评测，支持无参考答案评测；优化LLM裁判使用，预置“无参考答案直接打分” 和 “判断答案是否与参考答案一致”两种模式，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#qa)
- 🔥 **[2025.06.19]** 新增支持[BFCL-v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3)评测基准，用于评测模型在多种场景下的函数调用能力，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)。
- 🔥 **[2025.06.02]** 新增支持大海捞针测试（Needle-in-a-Haystack），指定`needle_haystack`即可进行测试，并在`outputs/reports`文件夹下生成对应的heatmap，直观展现模型性能，使用[参考](https://evalscope.readthedocs.io/zh-cn/latest/third_party/needle_haystack.html)。
- 🔥 **[2025.05.29]** 新增支持[DocMath](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary)和[FRAMES](https://modelscope.cn/datasets/iic/frames/summary)两个长文档评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.05.16]** 模型服务性能压测支持设置多种并发，并输出性能压测报告，[参考示例](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id3)。
- 🔥 **[2025.05.13]** 新增支持[ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static)数据集，评测模型的工具调用能力，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)；支持[DROP](https://modelscope.cn/datasets/AI-ModelScope/DROP/dataPeview)和[Winogrande](https://modelscope.cn/datasets/AI-ModelScope/winogrande_val)评测基准，评测模型的推理能力。
- 🔥 **[2025.04.29]** 新增Qwen3评测最佳实践，[欢迎阅读📖](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html)
- 🔥 **[2025.04.27]** 支持文生图评测：支持MPS、HPSv2.1Score等8个指标，支持EvalMuse、GenAI-Bench等评测基准，参考[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/aigc/t2i.html)
- 🔥 **[2025.04.10]** 模型服务压测工具支持`/v1/completions`端点（也是vLLM基准测试的默认端点）
- 🔥 **[2025.04.08]** 支持OpenAI API兼容的Embedding模型服务评测，查看[使用文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html#configure-evaluation-parameters)
- 🔥 **[2025.03.27]** 新增支持[AlpacaEval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview)和[ArenaHard](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)评测基准，使用注意事项请查看[文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)
- 🔥 **[2025.03.20]** 模型推理服务压测支持random生成指定范围长度的prompt，参考[使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random)
- 🔥 **[2025.03.13]** 新增支持[LiveCodeBench](https://www.modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary)代码评测基准，指定`live_code_bench`即可使用；支持QwQ-32B 在LiveCodeBench上评测，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.11]** 新增支持[SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)和[Chinese SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)评测基准，用与评测模型的事实正确性，指定`simple_qa`和`chinese_simpleqa`使用。同时支持指定裁判模型，参考[相关参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。
- 🔥 **[2025.03.07]** 新增QwQ-32B模型评测最佳实践，评测了模型的推理能力以及推理效率，参考[📖QwQ-32B模型评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html)。
- 🔥 **[2025.03.04]** 新增支持[SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary)数据集，其覆盖 13 个门类、72 个一级学科和 285 个二级学科，共 26529 个问题，指定`super_gpqa`即可使用。
- 🔥 **[2025.03.03]** 新增支持评测模型的智商和情商，参考[📖智商和情商评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/iquiz.html)，来测测你家的AI有多聪明？
- 🔥 **[2025.02.27]** 新增支持评测推理模型的思考效率，参考[📖思考效率评测最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/think_eval.html)，该实现参考了[Overthinking](https://doi.org/10.48550/arXiv.2412.21187) 和 [Underthinking](https://doi.org/10.48550/arXiv.2501.18585)两篇工作。
- 🔥 **[2025.02.25]** 新增支持[MuSR](https://modelscope.cn/datasets/AI-ModelScope/MuSR)和[ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary)两个模型推理相关评测基准，datasets分别指定`musr`和`process_bench`即可使用。
- 🔥 **[2025.02.18]** 支持AIME25数据集，包含15道题目（Grok3 在该数据集上得分为93分）
- 🔥 **[2025.02.13]** 支持DeepSeek蒸馏模型评测，包括AIME24 MATH-500 GPQA-Diamond数据集，参考[最佳实践](https://evalscope.readthedocs.io/zh-cn/latest/best_practice/deepseek_r1_distill.html)；支持指定`eval_batch_size`参数，加速模型评测
- 🔥 **[2025.01.20]** 支持可视化评测结果，包括单模型评测结果和多模型评测结果对比，参考[📖可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)；新增[`iquiz`](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)评测样例，评测模型的IQ和EQ。
- 🔥 **[2025.01.07]** Native backend: 支持模型API评测，参考[📖模型API评测指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#api)；新增支持`ifeval`评测基准。
- 🔥🔥 **[2024.12.31]** 支持基准评测添加，参考[📖基准评测添加指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)；支持自定义混合数据集评测，用更少的数据，更全面的评测模型，参考[📖混合数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html)
- 🔥 **[2024.12.13]** 模型评测优化，不再需要传递`--template-type`参数；支持`evalscope eval --args`启动评测，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html)
- 🔥 **[2024.11.26]** 模型推理压测工具重构完成：支持本地启动推理服务、支持Speed Benchmark；优化异步调用错误处理，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)
- 🔥 **[2024.10.31]** 多模态RAG评测最佳实践发布，参考[📖博客](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag)
- 🔥 **[2024.10.23]** 支持多模态RAG评测，包括[CLIP_Benchmark](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/clip_benchmark.html)评测图文检索器，以及扩展了[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)以支持端到端多模态指标评测。
- 🔥 **[2024.10.8]** 支持RAG评测，包括使用[MTEB/CMTEB](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html)进行embedding模型和reranker的独立评测，以及使用[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)进行端到端评测。
- 🔥 **[2024.09.18]** 我们的文档增加了博客模块，包含一些评测相关的技术调研和分享，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/blog/index.html)
- 🔥 **[2024.09.12]** 支持 LongWriter 评测，您可以使用基准测试 [LongBench-Write](evalscope/third_party/longbench_write/README.md) 来评测长输出的质量以及输出长度。
- 🔥 **[2024.08.30]** 支持自定义数据集评测，包括文本数据集和多模态图文数据集。
- 🔥 **[2024.08.20]** 更新了官方文档，包括快速上手、最佳实践和常见问题等，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/)。
- 🔥 **[2024.08.09]** 简化安装方式，支持pypi安装vlmeval相关依赖；优化多模态模型评测体验，基于OpenAI API方式的评测链路，最高加速10倍。
- 🔥 **[2024.07.31]** 重要修改：`llmuses`包名修改为`evalscope`，请同步修改您的代码。
- 🔥 **[2024.07.26]** 支持**VLMEvalKit**作为第三方评测框架，发起多模态模型评测任务。
- 🔥 **[2024.06.29]** 支持**OpenCompass**作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评测任务配置。
- 🔥 **[2024.06.13]** EvalScope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 。
- 🔥 **[2024.06.13]** 接入Agent评测集ToolBench。
</details>

## ❤️ 社区与支持

欢迎加入我们的社区，与其他开发者交流并获取帮助。

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  微信群 | 钉钉群
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/asset/discord_qr.jpg" width="160" height="160">  |  <img src="docs/asset/wechat.png" width="160" height="160"> | <img src="docs/asset/dingding.png" width="160" height="160">



## 🛠️ 环境准备

我们推荐使用 `conda` 创建虚拟环境，并使用 `pip` 安装。

1.  **创建并激活 Conda 环境** (推荐使用 Python 3.10)
    ```shell
    conda create -n evalscope python=3.10
    conda activate evalscope
    ```

2.  **安装 EvalScope**

    - **方式一：通过 PyPI 安装 (推荐)**
      ```shell
      pip install evalscope
      ```

    - **方式二：通过源码安装 (用于开发)**
      ```shell
      git clone https://github.com/modelscope/evalscope.git
      cd evalscope
      pip install -e .
      ```

3.  **安装额外依赖** (可选)
    根据您的需求，安装相应的功能扩展：
    ```shell
    # 性能测试
    pip install 'evalscope[perf]'

    # 可视化App
    pip install 'evalscope[app]'

    # 其他评测后端
    pip install 'evalscope[opencompass]'
    pip install 'evalscope[vlmeval]'
    pip install 'evalscope[rag]'

    # 安装所有依赖
    pip install 'evalscope[all]'
    ```
    > 如果您通过源码安装，请将 `evalscope` 替换为 `.`，例如 `pip install '.[perf]'`。

> [!NOTE]
> 本项目曾用名 `llmuses`。如果您需要使用 `v0.4.3` 或更早版本，请运行 `pip install llmuses<=0.4.3` 并使用 `from llmuses import ...` 导入。


## 🚀 快速开始

您可以通过**命令行**或 **Python 代码**两种方式启动评测任务。

### 方式1. 使用命令行

在任意路径下执行 `evalscope eval` 命令即可开始评测。以下命令将在 `gsm8k` 和 `arc` 数据集上评测 `Qwen/Qwen2.5-0.5B-Instruct` 模型，每个数据集只取 5 个样本。

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

### 方式2. 使用Python代码

使用 `run_task` 函数和 `TaskConfig` 对象来配置和启动评测任务。

```python
from evalscope import run_task TaskConfig

# 配置评测任务
task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct'
    datasets=['gsm8k' 'arc']
    limit=5
)

# 启动评测
run_task(task_cfg)
```

<details><summary><b>💡 提示：</b> `run_task` 还支持字典、YAML 或 JSON 文件作为配置。</summary>

**使用 Python 字典**

```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct'
    'datasets': ['gsm8k' 'arc']
    'limit': 5
}
run_task(task_cfg=task_cfg)
```

**使用 YAML 文件** (`config.yaml`)
```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
</details>

### 输出结果
评测完成后，您将在终端看到如下格式的报告：
```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

## 📈 进阶用法

### 自定义评测参数

您可以通过命令行参数精细化控制模型加载、推理和数据集配置。

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master" "precision": "torch.float16" "device_map": "auto"}' \
 --generation-config '{"do_sample":true"temperature":0.6"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0 "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

- `--model-args`: 模型加载参数，如 `revision` `precision` 等。
- `--generation-config`: 模型生成参数，如 `temperature` `max_tokens` 等。
- `--dataset-args`: 数据集配置参数，如 `few_shot_num` 等。

详情请参考 [📖 全部参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)。

### 评测在线模型 API

EvalScope 支持评测通过 API 部署的模型服务（如 vLLM 部署的服务）。只需指定服务地址和 API Key 即可。

1.  **启动模型服务** (以 vLLM 为例)
    ```shell
    export VLLM_USE_MODELSCOPE=True
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --served-model-name qwen2.5 \
      --port 8801
    ```

2.  **运行评测**
    ```shell
    evalscope eval \
     --model qwen2.5 \
     --eval-type openai_api \
     --api-url http://127.0.0.1:8801/v1 \
     --api-key EMPTY \
     --datasets gsm8k \
     --limit 10
    ```

### ⚔️ 竞技场模式 (Arena)

竞技场模式通过模型间的两两对战（Pairwise Battle）来评估模型性能，并给出胜率和排名，非常适合多模型横向对比。

```text
# 评测结果示例
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)
```
详情请参考 [📖 竞技场模式使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)。

### 🖊️ 自定义数据集评测

EvalScope 允许您轻松添加和评测自己的数据集。详情请参考 [📖 自定义数据集评测指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/index.html)。


## 🧪 其他评测后端
EvalScope 支持通过第三方评测框架（我们称之为“后端”）发起评测任务，以满足多样化的评测需求。

- **Native**: EvalScope 的默认评测框架，功能全面。
- **OpenCompass**: 专注于纯文本评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/opencompass_backend.html)
- **VLMEvalKit**: 专注于多模态评测。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**: 专注于 RAG 评测，支持 Embedding 和 Reranker 模型。 [📖 使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/index.html)
- **第三方评测工具**: 支持 [ToolBench](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html) 等评测任务。

## ⚡ 推理性能评测工具
EvalScope 提供了一个强大的压力测试工具，用于评估大语言模型服务的性能。

- **关键指标**: 支持吞吐量 (Tokens/s)、首字延迟 (TTFT)、Token 生成延迟 (TPOT) 等。
- **结果记录**: 支持将结果记录到 `wandb` 和 `swanlab`。
- **速度基准**: 可生成类似官方报告的速度基准测试结果。

详情请参考 [📖 性能测试使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/index.html)。

输出示例如下：
<p align="center">
    <img src="docs/zh/user_guides/stress_test/images/multi_perf.png" style="width: 80%;">
</p>


## 📊 可视化评测结果

EvalScope 提供了一个基于 Gradio 的 WebUI，用于交互式地分析和比较评测结果。

1.  **安装依赖**
    ```bash
    pip install 'evalscope[app]'
    ```

2.  **启动服务**
    ```bash
    evalscope app
    ```
    访问 `http://127.0.0.1:7861` 即可打开可视化界面。

<table>
  <tr>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/setting.png" alt="Setting" style="width: 90%;" />
      <p>设置界面</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/model_compare.png" alt="Model Compare" style="width: 100%;" />
      <p>模型比较</p>
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/report_overview.png" alt="Report Overview" style="width: 100%;" />
      <p>报告概览</p>
    </td>
    <td style="text-align: center;">
      <img src="docs/zh/get_started/images/report_details.png" alt="Report Details" style="width: 91%;" />
      <p>报告详情</p>
    </td>
  </tr>
</table>

详情请参考 [📖 可视化评测结果](https://evalscope.readthedocs.io/zh-cn/latest/get_started/visualization.html)。

## 👷‍♂️ 贡献

我们欢迎来自社区的任何贡献！如果您希望添加新的评测基准、模型或功能，请参考我们的 [贡献指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html)。

感谢所有为 EvalScope 做出贡献的开发者！

<a href="https://github.com/modelscope/evalscope/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=modelscope/evalscope"><br><br>
      </th>
    </tr>
  </table>
</a>


## 📚 引用

如果您在研究中使用了 EvalScope，请引用我们的工作：
```bibtex
@misc{evalscope_2024
    title={{EvalScope}: Evaluation Framework for Large Models}
    author={ModelScope Team}
    year={2024}
    url={https://github.com/modelscope/evalscope}
}
```


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)

# Arena Mode

Arena mode allows you to configure multiple candidate models and specify a baseline model. The evaluation is conducted through pairwise battles between each candidate model and the baseline model with the win rate and ranking of each model outputted at the end. This approach is suitable for comparative evaluation among multiple models and intuitively reflects the strengths and weaknesses of each model.

## Data Preparation

To support arena mode **all candidate models need to run inference on the same dataset**. The dataset can be a general QA dataset or a domain-specific one. Below is an example using a custom `general_qa` dataset. See the [documentation](../advanced_guides/custom_dataset/llm.md#question-answering-format-qa) for details on using this dataset.

The JSONL file for the `general_qa` dataset should be in the following format. Only the `query` field is required; no additional fields are necessary. Below are two example files:

- Example content of the `arena.jsonl` file:
    ```json
    {"query": "How can I improve my time management skills?"}
    {"query": "What are the most effective ways to deal with stress?"}
    {"query": "What are the main differences between Python and JavaScript programming languages?"}
    {"query": "How can I increase my productivity while working from home?"}
    {"query": "Can you explain the basics of quantum computing?"}
    ```

- Example content of the `example.jsonl` file (with reference answers):
    ```json
    {"query": "What is the capital of France?" "response": "The capital of France is Paris."}
    {"query": "What is the largest mammal in the world?" "response": "The largest mammal in the world is the blue whale."}
    {"query": "How does photosynthesis work?" "response": "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."}
    {"query": "What is the theory of relativity?" "response": "The theory of relativity developed by Albert Einstein describes the laws of physics in relation to observers in different frames of reference."}
    {"query": "Who wrote 'To Kill a Mockingbird'?" "response": "Harper Lee wrote 'To Kill a Mockingbird'."}
    ```

## Candidate Model Inference

After preparing the dataset you can use EvalScope's `run_task` method to perform inference with the candidate models and obtain their outputs for subsequent battles.

Below is an example of how to configure inference tasks for three candidate models: `Qwen2.5-0.5B-Instruct` `Qwen2.5-7B-Instruct` and `Qwen2.5-72B-Instruct` using the same configuration for inference.

Run the following code:
```python
import os
from evalscope import TaskConfig run_task
from evalscope.constants import EvalType

models = ['qwen2.5-72b-instruct' 'qwen2.5-7b-instruct' 'qwen2.5-0.5b-instruct']

task_list = [TaskConfig(
    model=model
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=os.getenv('DASHSCOPE_API_KEY')
    eval_type=EvalType.SERVICE
    datasets=[
        'general_qa'
    ]
    dataset_args={
        'general_qa': {
            'dataset_id': 'custom_eval/text/qa'
            'subset_list': [
                'arena'
                'example'
            ]
        }
    }
    eval_batch_size=10
    generation_config={
        'temperature': 0
        'n': 1
        'max_tokens': 4096
    }) for model in models]

run_task(task_cfg=task_list)
```

<details><summary>Click to view inference results</summary>

Since the `arena` subset does not have reference answers no evaluation metrics are available for this subset. The `example` subset has reference answers so evaluation metrics will be output.
```text
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| Model                 | Dataset    | Metric          | Subset   |   Num |   Score | Cat.0   |
+=======================+============+=================+==========+=======+=========+=========+
| qwen2.5-0.5b-instruct | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-R       | example  |    12 |  0.8611 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-P       | example  |    12 |  0.1341 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-1-F       | example  |    12 |  0.1983 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-R       | example  |    12 |  0.55   | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-P       | example  |    12 |  0.0404 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-2-F       | example  |    12 |  0.0716 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-R       | example  |    12 |  0.8611 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-P       | example  |    12 |  0.1193 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | Rouge-L-F       | example  |    12 |  0.1754 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-1          | example  |    12 |  0.1192 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-2          | example  |    12 |  0.0403 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-3          | example  |    12 |  0.0135 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-0.5b-instruct | general_qa | bleu-4          | example  |    12 |  0.0079 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-P       | example  |    12 |  0.1149 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-1-F       | example  |    12 |  0.1612 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-R       | example  |    12 |  0.6833 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-P       | example  |    12 |  0.0813 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-2-F       | example  |    12 |  0.1027 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-P       | example  |    12 |  0.101  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | Rouge-L-F       | example  |    12 |  0.1361 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-1          | example  |    12 |  0.1009 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-2          | example  |    12 |  0.0807 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-3          | example  |    12 |  0.0625 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-72b-instruct  | general_qa | bleu-4          | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | AverageAccuracy | arena    |    10 | -1      | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-P       | example  |    12 |  0.104  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-1-F       | example  |    12 |  0.1418 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-R       | example  |    12 |  0.7    | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-P       | example  |    12 |  0.078  | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-2-F       | example  |    12 |  0.0964 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-R       | example  |    12 |  0.9722 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-P       | example  |    12 |  0.0942 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | Rouge-L-F       | example  |    12 |  0.1235 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-1          | example  |    12 |  0.0939 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-2          | example  |    12 |  0.0777 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-3          | example  |    12 |  0.0625 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
| qwen2.5-7b-instruct   | general_qa | bleu-4          | example  |    12 |  0.0556 | default |
+-----------------------+------------+-----------------+----------+-------+---------+---------+
```
</details>

## Candidate Model Battles

Next you can use EvalScope's `general_arena` method to conduct battles among candidate models and get their win rates and rankings on each subset. To achieve robust automatic battles you need to configure an LLM as the judge that compares the outputs of models.

During evaluation EvalScope will automatically parse the public evaluation set of candidate models use the judge model to compare the output of each candidate model with the baseline and determine which is better (to avoid model bias outputs are swapped for two rounds per comparison). The judge model's outputs are parsed as win draw or loss and each candidate model's **Elo score** and **win rate** are calculated.

Run the following code:
```python
import os
from evalscope import TaskConfig run_task

task_cfg = TaskConfig(
    model_id='Arena'  # Model ID is 'Arena'; you can omit specifying model ID
    datasets=[
        'general_arena'  # Must be 'general_arena' indicating arena mode
    ]
    dataset_args={
        'general_arena': {
            # 'system_prompt': 'xxx' # Optional: customize the judge model's system prompt here
            # 'prompt_template': 'xxx' # Optional: customize the judge model's prompt template here
            'extra_params':{
                # Configure candidate model names and corresponding report paths
                # Report paths refer to the output paths from the previous step for parsing model inference results
                'models':[
                    {
                        'name': 'qwen2.5-0.5b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-0.5b-instruct'
                    }
                    {
                        'name': 'qwen2.5-7b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-7b-instruct'
                    }
                    {
                        'name': 'qwen2.5-72b'
                        'report_path': 'outputs/20250702_204346/reports/qwen2.5-72b-instruct'
                    }
                ]
                # Set baseline model must be one of the candidate models
                'baseline': 'qwen2.5-7b'
            }
        }
    }
    # Configure judge model parameters
    judge_model_args={
        'model_id': 'qwen-plus'
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        'api_key': os.getenv('DASHSCOPE_API_KEY')
        'generation_config': {
            'temperature': 0.0
            'max_tokens': 8000
        }
    }
    judge_worker_num=5
    # use_cache='outputs/xxx' # Optional: to add new candidate models to existing results specify the existing results path
)

run_task(task_cfg=task_cfg)
```

<details><summary>Click to view evaluation results</summary>

```text
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Model   | Dataset       | Metric        | Subset                                     |   Num |   Score | Cat.0   |
+=========+===============+===============+============================================+=======+=========+=========+
| Arena   | general_arena | winrate       | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0185 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.5469 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.075  | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.8382 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate       | OVERALL                                    |    44 |  0.3617 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0185 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.3906 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.025  | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.7276 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_lower | OVERALL                                    |    44 |  0.2826 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&example@qwen2.5-0.5b&qwen2.5-7b |    12 |  0.0909 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&example@qwen2.5-72b&qwen2.5-7b  |    12 |  0.6875 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&arena@qwen2.5-0.5b&qwen2.5-7b   |    10 |  0.0909 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | general_qa&arena@qwen2.5-72b&qwen2.5-7b    |    10 |  0.9412 | default |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+
| Arena   | general_arena | winrate_upper | OVERALL                                    |    44 |  0.4469 | -       |
+---------+---------------+---------------+--------------------------------------------+-------+---------+---------+ 
```
</details>


The automatically generated model leaderboard is as follows (output file located in `outputs/xxx/reports/Arena/leaderboard.txt`):

The leaderboard is sorted by win rate in descending order. As shown the `qwen2.5-72b` model performs best across all subsets with the highest win rate while the `qwen2.5-0.5b` model performs the worst.

```text
=== OVERALL LEADERBOARD ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)

=== DATASET LEADERBOARD: general_qa ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            69.3  (-13.3 / +12.2)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            4.7  (-2.5 / +4.4)

=== SUBSET LEADERBOARD: general_qa - example ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            54.7  (-15.6 / +14.1)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            1.8  (+0.0 / +7.2)

=== SUBSET LEADERBOARD: general_qa - arena ===
Model           WinRate (%)  CI (%)
------------  -------------  ---------------
qwen2.5-72b            83.8  (-11.1 / +10.3)
qwen2.5-7b             50    (+0.0 / +0.0)
qwen2.5-0.5b            7.5  (-5.0 / +1.6)
```

## Visualization of Battle Results

To intuitively display the results of the battles between candidate models and the baseline EvalScope provides a visualization feature allowing you to compare the results of each candidate model against the baseline model for each sample.

Run the command below to launch the visualization interface:
```shell
evalscope app
```
Open `http://localhost:7860` in your browser to view the visualization page.

Workflow:
1. Select the latest `general_arena` evaluation report and click the "Load and View" button.
2. Click dataset details and select the battle results between your candidate model and the baseline.
3. Adjust the threshold to filter battle results (normalized scores range from 0-1; 0.5 indicates a tie scores above 0.5 indicate the candidate is better than the baseline below 0.5 means worse).

Example below: a battle between `qwen2.5-72b` and `qwen2.5-7b`. The model judged the 72b as better:

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/arena_example.jpg)


# Sandbox Environment Usage

To complete LLM code capability evaluation we need to set up an independent evaluation environment to avoid executing erroneous code in the development environment and causing unavoidable losses. Currently EvalScope has integrated the [ms-enclave](https://github.com/modelscope/ms-enclave) sandbox environment allowing users to evaluate model code capabilities in a controlled environment such as using evaluation benchmarks like HumanEval and LiveCodeBench.

The following introduces two different sandbox usage methods:

- Local usage: Set up the sandbox environment on a local machine and conduct evaluation locally requiring Docker support on the local machine;
- Remote usage: Set up the sandbox environment on a remote server and conduct evaluation through API interfaces requiring Docker support on the remote machine.

## 1. Local Usage

Use Docker to set up a sandbox environment on a local machine and conduct evaluation locally requiring Docker support on the local machine.

### Environment Setup

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in your local Python environment:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration
When running evaluations add the `use_sandbox` and `sandbox_type` parameters to automatically enable the sandbox environment. Other parameters remain the same as regular evaluations:

Here's a complete example code for model evaluation on HumanEval:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig run_task

task_config = TaskConfig(
    model='qwen-plus'
    datasets=['humaneval']
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=env.get('DASHSCOPE_API_KEY')
    eval_type='openai_api'
    eval_batch_size=5
    limit=5
    generation_config={
        'max_tokens': 4096
        'temperature': 0.0
        'seed': 42
    }
    use_sandbox=True # enable sandbox
    sandbox_type='docker' # specify sandbox type
    judge_worker_num=5 # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation EvalScope will automatically start and manage the sandbox environment ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] Local sandbox manager started
...
```

## 2. Remote Usage

Set up the sandbox environment on a remote server and conduct evaluation through API interfaces requiring Docker support on the remote machine.

### Environment Setup

You need to install and configure separately on both the remote machine and local machine.

#### Remote Machine

The environment installation on the remote machine is similar to the local usage method described above:

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in remote Python environment:

```bash
pip install evalscope[sandbox]
```

3. **Start sandbox server**: Run the following command to start the sandbox server:

```bash
ms-enclave server --host 0.0.0.0 --port 1234
```

#### Local Machine

The local machine does not need Docker installation at this point but needs to install EvalScope:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration

When running evaluations add the `use_sandbox` parameter to automatically enable the sandbox environment and specify the remote sandbox server's API address in `sandbox_manager_config`:

Complete example code is as follows:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig run_task

task_config = TaskConfig(
    model='qwen-plus'
    datasets=['humaneval']
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    api_key=env.get('DASHSCOPE_API_KEY')
    eval_type='openai_api'
    eval_batch_size=5
    limit=5
    generation_config={
        'max_tokens': 4096
        'temperature': 0.0
        'seed': 42
    }
    use_sandbox=True # enable sandbox
    sandbox_type='docker' # specify sandbox type
    sandbox_manager_config={
        'base_url': 'http://<remote_host>:1234'  # remote sandbox manager URL
    }
    judge_worker_num=5 # specify number of sandbox workers during evaluation
)

run_task(task_config)
```

During model evaluation EvalScope will communicate with the remote sandbox server through API ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] HTTP sandbox manager started connected to http://<remote_host>:1234
...
```


# EvalScope Service Deployment

## Introduction

EvalScope service mode provides HTTP API-based evaluation and stress testing capabilities designed to address the following scenarios:

1. **Remote Invocation**: Support remote evaluation functionality through network without configuring complex evaluation environments locally
2. **Service Integration**: Easily integrate evaluation capabilities into existing workflows CI/CD pipelines or automated testing systems
3. **Multi-user Collaboration**: Support multiple users or systems calling the evaluation service simultaneously improving resource utilization
4. **Unified Management**: Centrally manage evaluation resources and configurations for easier maintenance and monitoring
5. **Flexible Deployment**: Can be deployed on dedicated servers or container environments decoupled from business systems

The Flask service encapsulates EvalScope's core evaluation (eval) and stress testing (perf) functionalities providing services through standard RESTful APIs making evaluation capabilities callable and integrable like other microservices.

## Features

- **Model Evaluation** (`/api/v1/eval`): Support evaluation of OpenAI API-compatible models
- **Performance Testing** (`/api/v1/perf`): Support performance benchmarking of OpenAI API-compatible models
- **Parameter Query**: Provide parameter description endpoints

## Environment Setup


### Full Installation (Recommended)

```bash
pip install evalscope[service]
```

### Development Environment Installation

```bash
# Clone repository
git clone https://github.com/modelscope/evalscope.git
cd evalscope

# Install development version with service
pip install -e '.[service]'
```

## Starting the Service

### Command Line Launch

```bash
# Use default configuration (host: 0.0.0.0 port: 9000)
evalscope service

# Custom host and port
evalscope service --host 127.0.0.1 --port 9000

# Enable debug mode
evalscope service --debug
```

### Python Code Launch

```python
from evalscope.service import run_service

# Start service
run_service(host='0.0.0.0' port=9000 debug=False)
```

## API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response Example:**
```json
{
  "status": "ok"
  "service": "evalscope"
  "timestamp": "2025-12-04T10:00:00"
}
```

### 2. Model Evaluation

```bash
POST /api/v1/eval
```

**Request Body Example:**
```json
{
  "model": "qwen-plus"
  "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
  "api_key": "your-api-key"
  "datasets": ["gsm8k" "iquiz"]
  "limit": 10
  "generation_config": {
    "temperature": 0.0
    "max_tokens": 2048
  }
}
```

**Required Parameters:**
- `model`: Model name
- `datasets`: List of datasets
- `api_url`: API endpoint URL (OpenAI-compatible)

**Optional Parameters:**
- `api_key`: API key (default: "EMPTY")
- `limit`: Evaluation sample quantity limit
- `eval_batch_size`: Batch size (default: 1)
- `generation_config`: Generation configuration
  - `temperature`: Temperature parameter (default: 0.0)
  - `max_tokens`: Maximum generation tokens (default: 2048)
  - `top_p`: Nucleus sampling parameter
  - `top_k`: Top-k sampling parameter
- `work_dir`: Output directory
- `debug`: Debug mode
- `seed`: Random seed (default: 42)

**Response Example:**
```json
{
  "status": "success"
  "message": "Evaluation completed"
  "result": {"...": "..."}
  "output_dir": "/path/to/outputs/20251204_100000"
}
```

### 3. Performance Testing

```bash
POST /api/v1/perf
```

**Request Body Example:**
```json
{
  "model": "qwen-plus"
  "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  "api": "openai"
  "api_key": "your-api-key"
  "number": 100
  "parallel": 10
  "dataset": "openqa"
  "max_tokens": 2048
  "temperature": 0.0
}
```

**Required Parameters:**
- `model`: Model name
- `url`: Complete API endpoint URL

**Optional Parameters:**
- `api`: API type (openai/dashscope/anthropic/gemini default: "openai")
- `api_key`: API key
- `number`: Total number of requests (default: 1000)
- `parallel`: Concurrency level (default: 1)
- `rate`: Requests per second limit (default: -1 unlimited)
- `dataset`: Dataset name (default: "openqa")
- `max_tokens`: Maximum generation tokens (default: 2048)
- `temperature`: Temperature parameter (default: 0.0)
- `stream`: Whether to use streaming output (default: true)
- `debug`: Debug mode

**Response Example:**
```json
{
  "status": "success"
  "message": "Performance test completed"
  "output_dir": "/path/to/outputs"
  "results": {
    "parallel_10_number_100": {
      "metrics": {"...": "..."}
      "percentiles": {"...": "..."}
    }
  }
}
```

### 4. Get Evaluation Parameter Description

```bash
GET /api/v1/eval/params
```

Returns descriptions of all parameters supported by the evaluation endpoint.

### 5. Get Performance Test Parameter Description

```bash
GET /api/v1/perf/params
```

Returns descriptions of all parameters supported by the performance test endpoint.

## Usage Examples

### Testing Evaluation Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "api_key": "your-api-key"
    "datasets": ["gsm8k"]
    "limit": 5
  }'
```

### Testing Performance Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/perf \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    "api": "openai"
    "number": 50
    "parallel": 5
  }'
```

### Using Python requests

```python
import requests

# Evaluation request
eval_response = requests.post(
    'http://localhost:9000/api/v1/eval'
    json={
        'model': 'qwen-plus'
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        'api_key': 'your-api-key'
        'datasets': ['gsm8k' 'iquiz']
        'limit': 10
        'generation_config': {
            'temperature': 0.0
            'max_tokens': 2048
        }
    }
)
print(eval_response.json())

# Performance test request
perf_response = requests.post(
    'http://localhost:9000/api/v1/perf'
    json={
        'model': 'qwen-plus'
        'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        'api': 'openai'
        'number': 100
        'parallel': 10
        'dataset': 'openqa'
    }
)
print(perf_response.json())
```

## Important Notes

1. **OpenAI API-Compatible Models Only**: This service is designed specifically for OpenAI API-compatible models
2. **Long-Running Tasks**: Evaluation and performance testing tasks may take considerable time. We recommend setting appropriate HTTP timeout values on the client side as the API calls are synchronous and will block until completion.
3. **Output Directory**: Evaluation results are saved in the configured `work_dir` default is `outputs/`
4. **Error Handling**: The service returns detailed error messages and stack traces (in debug mode)
5. **Resource Management**: Pay attention to concurrency settings during stress testing to avoid server overload

## Error Codes

- `400`: Invalid request parameters
- `404`: Endpoint not found
- `500`: Internal server error

## Example Scenarios

### Scenario 1: Quick Evaluation of Qwen Model

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "api_key": "sk-..."
    "datasets": ["gsm8k"]
    "limit": 100
  }'
```

### Scenario 2: Stress Testing Locally Deployed Model

```bash
curl -X POST http://localhost:9000/api/v1/perf \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5"
    "url": "http://localhost:8000/v1/chat/completions"
    "api": "openai"
    "number": 1000
    "parallel": 20
    "max_tokens": 2048
  }'
```

### Scenario 3: Multi-Dataset Evaluation

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus"
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "datasets": ["gsm8k" "iquiz" "ceval"]
    "limit": 50
    "eval_batch_size": 4
  }'
```