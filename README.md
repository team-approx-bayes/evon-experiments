# SOAP-Bubbles: Structured Weight Uncertainty for Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2606.23357-b31b1b.svg)](https://arxiv.org/abs/2606.23357)
[![Python Version](https://img.shields.io/badge/python-%3E%3D%203.11-blue.svg)](https://pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D%202.0.0-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](LICENSE)

> [!NOTE]
> The optimizer implementation itself lives in a separate repository: [team-approx-bayes/evon](https://github.com/team-approx-bayes/evon). 
> **We plan to release further improvements to the optimizer, which will only be available in the EVON repository.**

Here, we release the experiment code, scripts, and supporting utilities used for the [paper](https://arxiv.org/pdf/2606.23357). We include a frozen EVON implementation, as used for our experiments. 

> **SOAP-Bubbles: Structured Weight Uncertainty for Neural Networks**  
> _Adrian Robert Minut, Nico Daheim, Marco Miani, Mohammad Emtiyaz Khan, Wu Lin, Thomas Möllenhoff_  
> **ArXiv Paper**: [https://arxiv.org/abs/2606.23357](https://arxiv.org/abs/2606.23357)

---

## Repository Structure

The codebase is organized into several directories for different experiments, built on top of the core optimizer implementations:

*   **`modded-nanogpt/`**: Code for running the GPT-2 pretraining experiments. Adapted from [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
    *   `train_gpt2.py`: Main script for training GPT-2 models.
    *   `eval_checkpoint_mc_loss.py`: Evaluates training checkpoints using Monte Carlo loss to measure weight uncertainty.
*   **`Minimalist_LLM_Pretraining/`**: Code for Llama pretraining experiments. Adapted from [OptimAI-Lab/Minimalist_LLM_Pretraining](https://github.com/OptimAI-Lab/Minimalist_LLM_Pretraining).
    *   `torchrun_main_DDP.py`: Pretraining script using PyTorch Distributed Data Parallel (DDP).
    *   `eval_llama_checkpoint_mc_loss.py`: Evaluates Llama checkpoints via Monte Carlo loss.
*   **`clip-finetuning/`**: Code for fine-tuning CLIP visual encoders on image classification datasets. Adapted from [crisostomi/model-merging](https://github.com/crisostomi/model-merging).
    *   `finetune.py`: Standalone script for training and evaluation.
    *   `best_hparams/`: Contains optimal hyperparameters for each dataset/optimizer pair.
*   **`src/`**: Contains the source code for the core packages.
    *   `vonsoap/`: Main Python package containing the optimizer implementations (SOAP, EVON, IVON).
*   **`scripts/`**: Helper bash scripts for launching training and checkpoint evaluations (e.g., `gpt_speedrun_local.bash`, `llama_checkpoint_mc_eval_local.bash`).

---

## Citation

If you use EVON or the SOAP-Bubbles framework in your research, please cite our paper:

```bibtex
@misc{minut2026soapbubbles,
      title={SOAP-Bubbles: Structured Weight Uncertainty for Neural Networks}, 
      author={Adrian Robert Minut and Nico Daheim and Marco Miani and Mohammad Emtiyaz Khan and Wu Lin and Thomas M{"o}llenhoff},
      year={2026},
      eprint={2606.23357},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.23357}
}
```

---

## License

EVON is licensed under the GPLv3+ License. See [LICENSE](LICENSE) for details.
