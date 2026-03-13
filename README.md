# Mechanistic Interpretability of Fine-Tuning: Phase Transitions in Algorithmic Tasks

This repository contains experiments designed to investigate the mechanistic interpretability of fine-tuning, specifically addressing how internal representations and circuits change during the process. We use `TransformerLens` to build, train, and dissect these toy models.

## Experimental Setup

To cleanly observe fine-tuning dynamics, we use a toy model approach focusing on modular arithmetic where circuits are known to form clearly. This is particularly well-suited for studying phase transitions (grokking).

- **Task 1 (Pre-training)**: Modular addition `a + b (mod P)`
- **Task 2 (Fine-tuning)**: Modular subtraction `a - b (mod P)`
- **Tokens**: `0` to `P-1` representing numbers, `P` for `+`, `P+1` for `-`, and `P+2` for `=`.
- **Format**: `[a, op, b, =] -> target: c`.

**Model Architecture:**
We use a 1-layer, 4-head attention-only `HookedTransformer` with $d_{model}=128$. For the full-scale experiment, we set the prime modulus $P=113$. We apply a strong weight decay (1.0) during AdamW optimization to encourage structured, grokking-style representations.

## Training Pipeline

The training pipeline consists of two phases:
1.  **Pre-training**: The base model is trained to convergence on Task 1 (Addition) using 50% of the possible addition pairs as the train set, retaining the other 50% as a test set.
2.  **Fine-tuning**: The converged Task 1 model is further trained on Task 2 (Subtraction). Crucially, during fine-tuning, we track the loss and accuracy of *both* the Task 2 test set and the Task 1 holdout set.

## Results & Interpretations

### 1. Catastrophic Forgetting & Phase Transitions

During fine-tuning, we observe the dynamics of how the new task is learned and how the old task is forgotten.

![Phase Transition Plot](full_phase_transition.png)

**Interpretation:**
The plot demonstrates a clear phase transition. Initially, the model's accuracy on the new task (subtraction) remains near zero, while its accuracy on the pre-trained task (addition) stays perfectly intact. 

However, around step 400, a phase transition occurs. The loss for the fine-tuning task suddenly plummets, and its accuracy shoots up to ~100%. Simultaneously, the accuracy for the pre-training task catastrophically drops to near zero. This indicates that the parameters previously responsible for performing addition have been thoroughly overwritten or actively destroyed to accommodate the subtraction circuits, rather than the model finding a superposition of both skills. The model's limited capacity (1-layer) forces a hard choice between the two distributions.

### 2. Circuit Reorganization (SVD Analysis)

We can examine *how* the circuits changed by looking at the Singular Value Decomposition (SVD) of the OV (Output-Value) and QK (Query-Key) weight matrices for each of the 4 attention heads before and after fine-tuning.

*Note: View the generated `svd_L0H*.png` plots in the repository to see the shift in singular values.*

**Interpretation:**
The SVD comparisons show that the principal components of the weight matrices undergo significant shifts. Some heads show a flattening of their singular value spectrum in the OV circuit, suggesting they are changing from highly specialized rank-1/rank-2 feature extractors (useful for addition) into more distributed or entirely different feature representations (for subtraction). The QK circuits also shift, indicating that the heads are changing *what* they attend to (e.g., perhaps changing focus from the first operand to the second based on the operator token).

### 3. Circuit Analysis on Fine-Tuned Data

We analyze a specific example from the fine-tuning task: `[100, -, 20, =] -> target: 80`. We use two interpretability techniques to see which heads are contributing to the correct answer `80`.

**Direct Logit Attribution (DLA):**
DLA measures how much a specific head directly writes the direction of the target token's unembedding vector into the residual stream at the final sequence position.
*   **L0H0:** 0.023
*   **L0H1:** 0.013
*   **L0H2:** -0.022
*   **L0H3:** 0.011

**Interpretation:** Heads L0H0, L0H1, and L0H3 are positively contributing to the prediction of the correct subtraction answer, while Head L0H2 is actively pushing the prediction away from the correct answer.

**Causal Activation Patching:**
We took the Base Model (which only knows addition) and patched in the activation outputs ($z$) of individual heads from the Fine-Tuned Model (which knows subtraction) on the same prompt. We measured how much patching that specific head increased the probability of the correct subtraction answer (80).
*   **L0H0:** -0.000017
*   **L0H1:** +0.002698
*   **L0H2:** +0.000129
*   **L0H3:** +0.000211

**Interpretation:** 
Patching the output of Head L0H1 from the fine-tuned model into the base model caused the most significant increase in the correct token's probability. This causally links Head L0H1 as a primary driver of the newly learned subtraction algorithm. Interestingly, while L0H0 had the highest DLA, patching it into the base model had a slightly negative effect, suggesting its output relies on other coordinating features present in the fine-tuned model's residual stream that are missing in the base model.

## Running the Code

1.  `pip install -r requirements.txt` (requires PyTorch, TransformerLens, Matplotlib)
2.  Run the verification toy test: `python verify.py`
3.  Run the full analysis: `python run_full_analysis.py`
