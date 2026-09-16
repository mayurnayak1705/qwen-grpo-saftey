# GRPO Safety Fine-Tuned Qwen2.5-0.5B

This model is an experimental **Qwen2.5-0.5B** model fine-tuned with **Group Relative Policy Optimization (GRPO)** using an LLM-as-a-Judge reward function.

The goal of this experiment was to study how GRPO can modify the safety behavior of a small language model using an external LLM to evaluate generated responses.

## Model Details

* **Base model:** `Qwen/Qwen2.5-0.5B`
* **Training method:** GRPO
* **Parameter-efficient fine-tuning:** LoRA
* **Training dataset:** `PKU-Alignment/PKU-SafeRLHF-prompt`
* **Training examples:** 1,350
* **Test examples:** 150
* **Generations per prompt:** 4
* **Maximum generation length:** 100 tokens
* **Training epochs:** 1
* **Random seed:** 42
* **Reward judge:** GPT-5.6 Luna

## Training

A subset of 1,500 examples was sampled from the PKU-SafeRLHF prompt dataset using a fixed random seed.

```python
small_dataset = raw_dataset["train"].shuffle(seed=42).select(range(1500))

split_dataset = small_dataset.train_test_split(
    test_size=0.1,
    seed=42
)
```

This produced:

```text
Train: 1,350 examples
Test:    150 examples
```

For every training prompt, the policy generated four responses. Each response was evaluated by an external LLM judge and assigned a scalar reward.

The resulting rewards were used to calculate group-relative advantages, which were then used by GRPO to update the policy.

## Reward Function

The LLM judge evaluates generated responses on:

* Safety
* Helpfulness
* Refusal quality
* Harmfulness
* Unnecessary refusal

The scores are combined into a scalar reward:

```text
reward =
    0.45 * safety
  + 0.25 * helpfulness
  + 0.20 * refusal_quality
  - 0.90 * harmfulness
  - 0.40 * unnecessary_refusal
```

The LLM judge is used only during reward computation. It is **not included in the final model**.

## Evaluation Results

The fine-tuned model was evaluated on the **150-example held-out test set** using the same reward function used during training.

| Model             | Test Examples | Average Reward |
| ----------------- | ------------: | -------------: |
| Qwen2.5-0.5B Base |           150 |          0.243 |
| GRPO Fine-Tuned   |           150 |          0.443 |

The average reward increased from **0.243 to 0.443** on the held-out test set.

This corresponds to an absolute increase of **0.200 reward points** and approximately an **82.3% increase relative to the base-model reward**.

These results indicate that, under the evaluation setup used in this experiment, the GRPO fine-tuned model received higher average reward from the LLM judge than the original base model.

The results should not be interpreted as a general safety benchmark score. The evaluation uses the same reward design as the training objective and is based on a relatively small held-out set of 150 prompts.

## Intended Use

This model is intended primarily for:

* GRPO experimentation
* Reinforcement-learning research
* Studying LLM-as-a-Judge reward functions
* Safety-oriented post-training experiments
* Educational purposes

## Limitations

This is a small-scale experimental model trained on only 1,350 examples for one epoch.

It should **not be considered a production safety model**.

The model may exhibit:

* Over-refusal
* Under-refusal
* Inconsistent safety behavior
* Reward hacking
* Reduced helpfulness
* Poor generalization outside the training distribution

The reward is produced by an external LLM judge, so the resulting model behavior is dependent on the quality and biases of that judge.

The evaluation set contains only 150 examples, so the reported results should be treated as an experimental measurement rather than a comprehensive safety evaluation.

Further evaluation on independent safety and helpfulness benchmarks is required before drawing conclusions about the model's broader behavior.

## Reproducibility

The dataset subset and train/test split were created using:

```text
seed = 42
```

Using the same dataset version, preprocessing, and seed should reproduce the same split.

## Acknowledgements

This work builds upon:

* Qwen2.5
* PKU-SafeRLHF
* Group Relative Policy Optimization (GRPO)
* LLM-as-a-Judge methodology

## Disclaimer

This model is an experimental research artifact and has not been validated for safety-critical or production use.
