# ConfAgents: A Conformal-Guided Multi-Agent Framework for Cost-Efficient Medical Diagnosis

ConfAgents is our proposed adaptive multi-agent framework. This project aims to enhance the efficiency and robustness of complex problem-solving by intelligently allocating collaborative resources through a principled, two-stage process.

## **Key Features**

1. **Adaptive Multi-Agent Framework**: ConfAgents introduces a general framework that optimizes decision accuracy and computational resource allocation by enabling agents to work together adaptively.
2. **Confidence-Based Triage Mechanism**: The core of the framework is an innovative two-stage process. It first employs a reliable, confidence-based triage mechanism to evaluate outcomes from individual agents, escalating only the most complex and uncertain cases for collaborative deliberation.
3. **Dynamic Knowledge Integration**: For cases escalated by the triage system, ConfAgents initiates an enhanced collaborative process. During this process, agents can dynamically retrieve and integrate external knowledge to overcome the limitations of their static training data.
4. **Improved Efficiency and Robustness**: By collaborating only on difficult tasks, ConfAgents significantly reduces the computational overhead and latency from unnecessary collaboration, creating a more efficient and robust system. 

## Environmental Setups

- Create an environment `confagents` and activate it.

```bash
conda create -n confagents python=3.8
conda activate confagents
```

- Install the required packages.

```bash
pip install -r requirements.txt
```

## Usage

### Running LLM-based Multi-Agent Collaboration

- Run the following command to start the multi-agent collaboration framework as an example.

```bash
python conformal_pipeline.py --dataset='MedQA' --model='gpt-4o' --alpha=0.2
```

> The results can be found in the `output` directory.

## Datasets and Tasks

### Datasets

ConfAgents has been evaluated on the following datasets. 

- MedQA
- MMLU
- MedBullets
- AfriMedQA

We refer to the following paper for the selection of data samples.

Tang X, Shao D, Sohn J, et al. Medagentsbench: Benchmarking thinking models and agent frameworks for complex medical reasoning[J]. arXiv preprint arXiv:2503.07459, 2025.

### Tasks

ConfAgents has been evaluated on the multiple choices medical diagnosis tasks.