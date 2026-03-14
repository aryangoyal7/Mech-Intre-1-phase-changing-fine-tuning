# Inside the Black Box: How Fine-Tuning Rewires AI (A Mechanistic Interpretability Study)

*What actually happens inside a neural network when you fine-tune it? Does it politely learn a new skill alongside the old one, or does a violent, structural coup take place in its parameters? Let's crack open a Transformer and find out.*

---

## Introduction: Peeking Under the Hood

Large Language Models (LLMs) are notorious for being black boxes. We feed them data, they perform gradient descent, and somehow, they emerge with the ability to write code, compose poetry, or solve math problems. But understanding *how* they do this—the internal algorithms they learn—remains one of the most exciting frontiers in AI safety and research.

This field is called **Mechanistic Interpretability**. By reverse-engineering neural networks, we aim to translate the alien, distributed matrices of weights into human-understandable circuits.

Recently, I conducted an experiment to answer a seemingly simple question: **What happens to these internal circuits when we fine-tune a model?**

Our guiding questions were:
1. What goes on internally when transitioning from Task 1 (pre-training) to Task 2 (fine-tuning)? 
2. Does the model catastrophically forget the original distribution?
3. How do the circuits mapping to fine-tuned behavior differ from basic ones?
4. Do we see sudden "phase transitions" (like grokking) during this process?

## Background: Grokking and the Fourier Connection

To understand our fine-tuning results, we first have to talk about **grokking**. In a seminal [Alignment Forum post and paper](https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking) by Neel Nanda et al., the authors analyzed a small Transformer trained on modular addition ($a + b \pmod{113}$). 

They found a fascinating dynamic:
*   **Memorization Phase:** Initially, the model simply memorizes the training data. The training loss drops to near-zero, but the test loss remains high.
*   **Grokking Phase (Phase Transition):** Suddenly, long after training loss has bottomed out, the test loss plummets. The model "groks" the generalizing algorithm. 
*   **The Algorithm:** Mechanistically, the model learns to map the numbers into a circular, frequency-based representation using Discrete Fourier Transforms, exploiting trigonometric identities to perform addition! 
*   The transition is driven by weight decay, which heavily penalizes the high-norm memorization circuits, forcing the model to adopt the more parameter-efficient, generalizing Fourier circuit.

With this expected behavior as our baseline—knowing that toy mathematical models use highly structured, frequency-based circuits to generalize—we set out to see what happens when we *force the model to change its mind*.

## The Experiment: Forcing a Coup

To cleanly observe the fine-tuning dynamics, we used a constrained, toy model setup, utilizing the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library.

*   **The Model:** A tiny, 1-layer, 4-head attention-only HookedTransformer ($d_{model} = 128$).
*   **Task 1 (Pre-training):** Modular Addition: `[a, +, b, =] -> target: (a + b) % 113`
*   **Task 2 (Fine-Tuning):** Modular Subtraction: `[a, -, b, =] -> target: (a - b) % 113`

We set a high weight decay to encourage the clean, "grokking-style" representations. We trained the model to convergence on addition, and then slammed it with subtraction, recording every change along the way.

## Result 1: The Catastrophic Phase Transition

What happens to the model's performance during fine-tuning? Does it slowly fade away? 

Not exactly. We found a striking, sudden phase transition.

![Phase Transition Plot](https://raw.githubusercontent.com/aryangoyal7/Mech-Intre-1-phase-changing-fine-tuning/main/checkpoints/full_finetune/full_phase_transition.png)
*(Note: Visual of the phase transition plotting Task 1 vs Task 2 loss)*

1.  **The Stutter (Steps 0 - 300):** For the first few hundred steps of subtraction training, the new task's accuracy stays near 0%. During this time, the model stubbornly holds onto its addition algorithm—accuracy on the pre-trained task remains near perfect. It is resisting the new data.
2.  **The Phase Transition (Step ~400):** A sudden, non-linear shift occurs. The subtraction loss plummets, and accuracy hits 100%. Simultaneously, the addition accuracy drops to near 0%.

**The Takeaway:** In a constrained-capacity model, fine-tuning an opposing task doesn't result in "superposition" (sharing parameters cleanly between the two). It results in a violent coup. The exact moment the new circuit "clicks" into place is the exact moment the old circuit is structurally annihilated.

## Result 2: Reorganizing the Circuitry

To prove the old circuit was annihilated rather than bypassed, we mathematically sliced the model's "brain." 

We performed **Singular Value Decomposition (SVD)** on the attention heads' Output-Value (OV) and Query-Key (QK) weight matrices, both *before* and *after* fine-tuning. The OV circuit determines *what* information is moved, and the QK circuit determines *where* attention is paid.

If you look at the SVD plots in our [repository](https://github.com/aryangoyal7/Mech-Intre-1-phase-changing-fine-tuning), a pattern emerges: the leading singular values of several heads dramatically flatten or drop. 
*   **In grokking models**, a few massive singular values usually indicate a highly specialized rank-1/rank-2 feature extractor picking up specific Fourier frequencies. 
*   **During fine-tuning**, the model globally destroys these cleanly separated matrices, repurposing the attention heads' rank to capture the inverted frequencies needed for subtraction. *Every single head* was fundamentally rewritten.

## Result 3: Finding the True Driver (DLA vs. Patching)

How is the model specifically solving the subtraction problem now? Let's take the sequence `[100, -, 20, =] -> 80`. Which of our four attention heads (L0H0, L0H1, L0H2, L0H3) is responsible for predicting the `80`?

We applied two mechanistic interpretability metrics to find out. The results were a massive warning against trusting simple correlations.

### Metric A: Direct Logit Attribution (DLA)
DLA measures how much a specific head *directly writes* mathematically in the direction of the correct answer's logit scalar in the residual stream. 
*   **L0H0:** +0.023 (Strongest positive)
*   **L0H1:** +0.013 (Moderate positive)
*   ...

If we stopped here, we would say **L0H0** is the VIP solving the subtraction task. 

### Metric B: Causal Activation Patching
Patching is the gold standard for causality. We took the Base Model (which only knows addition) and surgically swapped in the output activations ($z$) of individual heads from the Fine-Tuned Model. If the head truly contains the standalone algorithm for subtraction, patching it in should increase the base model's probability of outputting `80`.

*   **Patching L0H0:** -0.000017 (Slightly negative effect)
*   **Patching L0H1:** +0.002698 (Massive positive spike)

**The Takeaway:** DLA lied to us! While L0H0's output aligned well with the final logit, patching proved it was completely useless on its own—it likely relies on other contextual features present in the fine-tuned model's residual stream. **L0H1**, on the other hand, was the true autonomous driver. It had learned the self-contained causal algorithm for subtraction, capable of hijacking the naive base model.

## Conclusion

Mechanistically analyzing fine-tuning gives us a visceral look at how AI actually learns. 
1. Models don't gracefully transition; they undergo sudden, catastrophic phase shifts where old algorithms are dismantled to make way for the new. 
2. The Fourier-based structures discovered in pure grokking models can be globally re-aligned into new frequencies when optimized under pressure. 
3. Finally, when trying to understand *how* a model is doing what it's doing, correlation metrics like DLA can be highly deceptive compared to causal intervention techniques like Activation Patching.

As we fine-tune massive frontiers models—trying to align them or teach them to use tools—it's worth remembering that we aren't just adjusting parameters. We are inducing structural coups in their internal circuitry.

---

*Code and technical breakdowns for these experiments are available on GitHub: [aryangoyal7/Mech-Intre-1-phase-changing-fine-tuning](https://github.com/aryangoyal7/Mech-Intre-1-phase-changing-fine-tuning).*
