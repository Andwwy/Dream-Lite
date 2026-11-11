# The SLEEP framework: technical guide

A walkthrough of Google's "sleep" method for continual learning in LLMs, how each stage works, the paper behind each stage, and the code you can reuse for Dream-Lite.

> **Sources.** Read on 2026-11-10 from the blog post, the arXiv HTML (v2) of the main paper, and the arXiv HTML of the Hope paper. Wherever a paper leaves something out, this guide says so. Text marked *[inferred]* is my reading, not the authors' wording.

---

## 0. The core sources

| What | Link | Code |
|---|---|---|
| **Main paper:** *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*. Behrouz, Hashemi, Javanmard, Mirrokni (Google / Cornell). v1 2 Jun 2026, v2 10 Jul 2026 | [arXiv 2606.03979](https://arxiv.org/abs/2606.03979), [HTML](https://arxiv.org/html/2606.03979) | **None released** (checked the abstract page, the paper body, and GitHub search) |
| **Blog:** *The SLEEP framework* (27 Oct 2026) | [research.google](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/) | none |
| **Architecture it builds on:** *Nested Learning: The Illusion of Deep Learning Architectures* (introduces Hope and CMS). Behrouz, Razaviyayn, Zhong, Mirrokni. NeurIPS 2025 | [arXiv 2512.24695](https://arxiv.org/abs/2512.24695), [blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) | No official code. Community versions exist (unverified): [obekt/HOPE-nested-learning](https://github.com/obekt/HOPE-nested-learning), [aryateja2106/nested-learning](https://github.com/aryateja2106/nested-learning) |

**Don't confuse it with** *Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference* (Lee, McLeish, Goldstein, Fanti, [arXiv 2605.26099](https://arxiv.org/abs/2605.26099)). That is a different method (extra offline recurrence passes at the points where context is evicted).

"SLEEP" is never expanded as an acronym. The paper says "sleep framework" or "sleep paradigm".

---

## 1. The big picture in one diagram

```
                ┌──────────────── WAKE ────────────────┐
 input stream → │ attention + high-frequency memory    │  (short-term; updated online)
                └──────────────────┬───────────────────┘
                                   │ every C^(ℓ*-1) steps
                ┌──────────────── SLEEP ───────────────┐
                │ Stage 1  MEMORY CONSOLIDATION (NREM) │
                │   1a grow: add low-rank expert {A,B} │
                │       to the slower block ℓ*         │
                │   1b Knowledge Seeding:              │
                │       GKD distillation + LTI reward  │
                │   1c prune: reset fast block's       │
                │       old experts                    │
                │ Stage 2  DREAMING (REM)              │
                │   2a sample dreams (random expert)   │
                │   2b gradient-based selection        │
                │   2c LoRA-SFT on each dream          │
                │   2d keep dreams that improve the    │
                │       task metric (ReST-EM)          │
                └──────────────────────────────────────┘
```

**The core idea.** Weights are arranged by how often they update. Fast weights learn quickly but forget quickly; slow weights hold knowledge. Sleep moves what the fast weights learned into slow weights, which first get new capacity. Then the model practices on its own synthetic data ("dreams") and keeps only the practice that measurably helps.

---

## 2. Background: Nested Learning, CMS, Hope

*Source: [arXiv 2512.24695](https://arxiv.org/abs/2512.24695), Sec. 3, 7, 8.*

### 2.1 Update frequency defines a "level"
- **Definition (update frequency):** $f_A$ is the number of updates component $A$ receives per unit time, where one data point is one time unit. Components that update at the same rate form one **level**. A higher level means a lower frequency.
- From this view, a standard Transformer has two extreme levels. Attention updates on every token (frequency ∞), and the MLP weights are fixed after pre-training (frequency 0).
- Optimizers and attention are both **associative memories**: $\mathcal{M}^* = \arg\min_{\mathcal{M}} \tilde{\mathcal L}(\mathcal M(\mathcal K);\mathcal V)$. Momentum SGD is a two-level memory that compresses past gradients, and linear attention is $\mathcal M_t = \mathcal M_{t-1} + v_t k_t^\top$.

### 2.2 Continuum Memory System (CMS)
CMS fills the gap between frequency ∞ and frequency 0 with a chain of MLP blocks, each updating at its own frequency:

$$y_t=\texttt{MLP}^{(f_k)}\big(\texttt{MLP}^{(f_{k-1})}(\cdots\texttt{MLP}^{(f_1)}(x_t))\big),\qquad C^{(\ell)}=\frac{\max_i C^{(i)}}{f_\ell}$$

Block $\ell$ changes only once every $C^{(\ell)}$ steps, applying the gradients it accumulated over that chunk:

$$\theta^{(f_\ell)}_{i+1}=\theta^{(f_\ell)}_i-\begin{cases}\sum_{t=i-C^{(\ell)}}^{i}\eta^{(\ell)}_t\, f(\theta^{(f_\ell)}_t;x_t) & i\equiv 0 \pmod{C^{(\ell)}}\\ 0 & \text{otherwise}\end{cases}$$

- **High frequency** means a small chunk: the block adapts fast and forgets fast. **Low frequency** means a large chunk: knowledge persists. Pre-training is the lowest level.
- A plain Transformer MLP is the special case $k=1$, $f=0$.
- **Starting from a pretrained model:** each CMS block is initialized from the pretrained MLP weights, $\texttt{MLP}^{(f_i)}_0 = \texttt{MLP}_{\text{pre-trained}_i}$, with a small internal learning rate (Sec. 7.3). *[inferred]* This is presumably how SLEEP is put on top of Qwen3 or Llama, but the SLEEP paper does not say so.

### 2.3 Hope
Hope = **self-modifying Titans**, followed by **CMS**. Every projection ($k, v, q, \eta, \alpha$, memory) is a small memory that learns in context and produces its own target values:

$$\mathcal M_{\square,t}=\mathcal M_{\square,t-1}(\alpha_t I-\eta_t k_t k_t^\top)-\eta_t\nabla\mathcal L(\mathcal M_{\square,t-1};k_t,\hat v_{\square,t}),\qquad \hat v_{\square,t}=\mathcal M_{\square,t-1}(v_t)$$

The output $o_t=\mathcal M_{\text{memory},t-1}(q_t)$ is then fed through the CMS chain. **Hope-Attention** replaces the Titans part with softmax attention.

Ancestor: **Titans** ([arXiv 2501.00663](https://arxiv.org/abs/2501.00663), [blog](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/); unofficial code: [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch)).

> **What SLEEP adds.** Nested Learning already moves knowledge from fast to slow blocks *online* during wake. SLEEP adds an *offline* phase that does this more deliberately (Sec. 2.2 of the main paper says the online part is not its contribution).

---

## 3. Wake phase and the sleep trigger

- **Short-term memory** = the sequence model (attention or a recurrent memory) plus the high-frequency CMS block(s) and the low-rank experts they have gathered. It is *not* the KV cache that gets consolidated; it is the weights of the fast block $\ell^*-1$.
- **When sleep runs:** when the step count is divisible by $C^{(\ell^*-1)}$. The full set of consolidation steps is $\{C^{(1)}b,\dots,C^{(k)}b\}$ for $b\in\mathbb N$. Each sleep consolidates block $\ell^*-1$ (sender) into block $\ell^*$ (receiver).

---

## 4. Stage 1: memory consolidation ("Knowledge Seeding")

*Main paper Sec. 3.2–3.3.*

### 4.1 Grow: parameter expansion
- Each CMS block is a **sparse MoE** with $s_\ell$ experts. Each sleep adds **one low-rank expert** to the receiving block:
  $$A^{(f_{\ell^*})}\in\mathbb R^{d\times d_{\text{low}}},\quad B^{(f_{\ell^*})}\in\mathbb R^{d_{\text{low}}\times d},\quad d_{\text{low}}\ll d$$
- In practice the parameters are **pre-allocated and masked** in the forward and backward passes until they are switched on. That is how the blog can say the architecture is "fixed" while capacity grows.
- Ancestors: Mixture of LoRA Experts ([arXiv 2404.13628](https://arxiv.org/abs/2404.13628)), MixLoRA ([arXiv 2404.15159](https://arxiv.org/abs/2404.15159)), LoRA ([arXiv 2106.09685](https://arxiv.org/abs/2106.09685)).
- **Not stated:** $d_{\text{low}}$, expert count, top-k routing, and which Qwen layers receive experts.

### 4.2 The three-step protocol: compute, consolidate, update
1. **Compute.** Work out the sender's pending base-weight update $\theta_{\text{old}}\to\theta_{\text{new}}$ from its accumulated gradients, but don't apply it yet.
2. **Consolidate.**
   - **Teacher** = the model before the update (old base weights plus all of the sender's accumulated experts).
   - **Student** = $\theta_{\text{new}}$, with the sender's experts reset, plus the new $\{A,B\}$.
   - **All student parameters are frozen except $\{A,B\}$.**
3. **Update.** Apply $\theta_{\text{new}}$, reset the sender's old experts (**synaptic pruning**, after Li et al. 2017), and switch on $\{A,B\}$ in block $\ell^*$.

**Why it works this way.** The knowledge held in the fast block's temporary experts is about to be erased. Before that happens, a small new slow-block expert is trained so the post-reset model behaves like the pre-reset model.

### 4.3 Distillation signal: GKD (on-policy distillation)
The transfer data $\mathcal D$ is **sampled from the teacher**. A fraction $\lambda$ of training uses **student-generated** sequences, scored by the teacher; gradients do not flow through the sampling step.

$$\mathcal L(\theta,\theta_{\text{exp}})=(1-\lambda)\,\mathbb E_{(x,y)\sim\mathcal D}\big[\mathcal F(\texttt{LM}_\theta\|\texttt{LM}_{\theta_{\text{exp}}})(y|x)\big]+\lambda\,\mathbb E_{x\sim\mathcal D}\,\mathbb E_{y\sim\texttt{LM}_{\theta_{\text{exp}}}(\cdot|x)}\big[\mathcal F(\cdot\|\cdot)(y|x)\big]$$

- Teacher = $\texttt{LM}_\theta$, the first argument. Student = $\texttt{LM}_{\theta_{\text{exp}}}$.
- **The divergence $\mathcal F$ is not named.** The GKD paper allows forward KL, reverse KL, or generalized JSD.
- Paper: **GKD**, *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (Agarwal et al., ICLR 2024, [arXiv 2306.13649](https://arxiv.org/abs/2306.13649)).
- **Code:** TRL's [`GKDTrainer`](https://huggingface.co/docs/trl/en/gkd_trainer). `lmbda` is the on-policy fraction, which plays the role of λ. `beta` interpolates the generalized JSD. `seq_kd` switches to sequence-level KD.

### 4.4 Learning to imitate (LTI): an RL reward on top of distillation
Take a **prefix** of a teacher dream $d^{(i)}$ and have the student complete it, giving $\hat d^{(i)}$. The reward is:

$$r=\gamma\, r_{\text{sem}}+(1-\gamma)\, r_{\text{abs}},\qquad r_{\text{abs}}=\begin{cases}1-\dfrac{z(\hat d,d)}{\max(|\hat d|,|d|)} & z\le z_0\\[2pt] 0&\text{otherwise}\end{cases}$$

- $z$ = Levenshtein (edit) distance; $z_0$ = a threshold.
- $r_{\text{sem}}\in\{0,1\}$: a **frozen reward model** decides whether the two texts mean the same thing.
- **Not stated:** which reward model, the value of $z_0$, and which RL estimator (GRPO? REINFORCE?).

### 4.5 Combined Knowledge-Seeding objective (as written)
$$\mathcal L_{\text{KS}}=\mathbb E_{x\sim\mathcal D}\Big[(1-\alpha)\,\mathbb E_{y\sim\texttt{LM}_{\theta_{\text{exp}}}}[r(y)]-\alpha\,\mathbb E_{y\sim\texttt{LM}_{\theta_{\text{exp}}}}[\mathcal D(\texttt{LM}_\theta\|\texttt{LM}_{\theta_{\text{exp}}})(y|x)]\Big]$$

⚠️ As written, this is something to **maximize** (reward minus divergence). Yet the text says $\{A,B\}$ are trained to "minimize the distillation loss". *[inferred]* The intended training loss is $-(1-\alpha)\mathbb E[r]+\alpha\,\mathbb E[\mathcal D]$.

### 4.6 Inconsistency to know about
- The contributions list says the teacher is *"a smaller version of model with (compressed) privilege information"*. That sounds like **context distillation**, where the teacher sees the context and the student doesn't.
- Sec. 3.3 instead defines the teacher as the **pre-update model**.
- The loss as written does not condition the teacher on any context.

Related context-distillation line: Cartridges ([arXiv 2506.06266](https://arxiv.org/abs/2506.06266), code [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges)), deep context distillation ([2503.08727](https://arxiv.org/abs/2503.08727)), prompt distillation ([2412.14964](https://arxiv.org/abs/2412.14964)), on-policy context distillation ([2602.12275](https://arxiv.org/abs/2602.12275)).

### 4.7 Synaptic pruning
After consolidation, every low-rank expert previously added to the sender block is reset, which frees its capacity for the next wake period.

---

## 5. Stage 2: dreaming (self-improvement)

*Main paper Sec. 3.4. This is SEAL's loop with two additions.*

| Step | Mechanism | Ancestor and code |
|---|---|---|
| **2a Generate** | For a task $(C,\tau)$ (context, metric), sample $m$ dreams: $\{\texttt{DREAM}^{(i)}\}\sim\texttt{LM}_\theta(\cdot\mid C)$. **New:** in each MoE block the router also turns on one **random expert** during sampling, which mixes in unrelated knowledge. | SEAL self-edits |
| **2b Select** | Score each dream with its SFT gradient $g^{(i)}=\nabla_\theta\mathcal L_{\text{SFT}}(\texttt{DREAM}^{(i)},\theta)$. Keep the **top-k plus b random** dreams. **New** relative to SEAL. | GREATS ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed165f2ff227cf36c7e3ef88957dadd9-Abstract.html), code [Jiachen-T-Wang/GREATS](https://github.com/Jiachen-T-Wang/GREATS)); G-DIG ([arXiv 2405.12915](https://arxiv.org/abs/2405.12915), code [xypan0/G-DIG](https://github.com/xypan0/G-DIG)) |
| **2c Update** | Train a separate copy on each kept dream with LoRA SFT: $\theta'^{(i)}\leftarrow\text{SFT}(\theta^{(i)},\texttt{DREAM}^{(i)})$. | SEAL inner loop |
| **2d Reinforce** | Binary reward: $r=1$ if the metric $\tau$ improves over the pre-dream model, otherwise 0. Optimized with **ReST-EM**, i.e. SFT on the dreams that earned reward. | ReST-EM, *Beyond Human Data* (Singh et al., [arXiv 2312.06585](https://arxiv.org/abs/2312.06585)); SEAL ([arXiv 2506.10943](https://arxiv.org/abs/2506.10943), code [Continual-Intelligence/SEAL](https://github.com/Continual-Intelligence/SEAL)) |

**Underspecified:**
- How the gradient vector $g^{(i)}$ becomes a scalar score (its norm? alignment with a target gradient?). GREATS scores alignment with a validation gradient, and G-DIG uses influence relative to a seed set plus clustering of gradients.
- The values of $k$, $b$, $m$.
- The prompt template used to generate dreams.

Blog wording vs. paper: the blog calls 2d "RL refinement" and describes it as SFT on top-rated dreams. Both are accurate, because ReST-EM *is* filtered SFT.

---

## 6. End-to-end pseudocode *[inferred from Sec. 3; not official]*

```python
for step, x in enumerate(stream):                       # WAKE
    y = model(x)                                        # attention + CMS forward
    accumulate_grads(model.cms, x)                      # CMS blocks update per chunk size
    for l in levels:
        if step % C[l] == 0:
            sleep(model, sender=l, receiver=l + 1)

def sleep(model, sender, receiver):
    # ---- Stage 1: consolidation ----
    theta_new = pending_update(model.cms[sender])        # compute, don't apply
    teacher   = frozen_copy(model)                       # old weights + sender experts
    student   = copy(model).apply(theta_new).reset_experts(sender)
    A, B      = student.cms[receiver].unmask_new_lowrank_expert()
    freeze(student, except_=[A, B])
    D = sample(teacher, prompts)                         # teacher-generated data
    for batch in D:
        loss = alpha * gkd(teacher, student, batch, lmbda)          # on-policy distill
        loss += -(1 - alpha) * lti_reward(student, teacher_prefixes) # RL imitation
        loss.backward(); opt.step()                                  # only A, B move
    model = student                                       # update + synaptic pruning

    # ---- Stage 2: dreaming ----
    dreams = [sample(model, C, random_expert=True) for _ in range(m)]
    kept   = topk(dreams, score=grad_score) + random.sample(dreams, b)
    good   = [d for d in kept if metric(lora_sft(model, d)) > metric(model)]
    model  = sft(model, good)                             # ReST-EM step
```

---

## 7. Results (verbatim from the paper's tables)

**SQuAD knowledge incorporation** (no-context accuracy; 974 questions; set up to follow SEAL "including models and parameters"; base model and grader **not named**):

| Method | single passage (n=1) | CPT (n=200) |
|---|---|---|
| Base | 31.9 | 31.9 |
| Fine-tuned, no dreaming | 33.4 | 32.0 |
| SEAL | 46.7 | 43.2 |
| **Sleep (Transformer + 4-level)** | **48.9** | **46.2** |
| w/o gradient selection | 47.1 | 45.2 |
| w/o random expert | 48.0 | 44.7 |
| **w/o dreaming** | **35.7** | **36.2** |

Five dreams are sampled per passage. "CPT" means one continued-pretraining run over all 200 passages; it is **not** a sequential continual-learning stream.

**Math** (average@16, AIME-24 / AIME-25 / HMMT-25):

| Model | Base | SFT | GRPO | OPSD | Sleep |
|---|---|---|---|---|---|
| Qwen3-1.7B | 49.8/34.5/25.7 | 47.3/36.1/22.9 | 51.0/38.6/26.1 | 51.6/40.0/28.1 | **53.2/40.2/29.3** |
| Qwen3-8B | 73.8/68.1/42.4 | 75.5/66.4/43.7 | 76.4/68.1/44.9 | 76.6/67.4/45.1 | **79.2/69.0/46.1** |

**Ablation on Qwen3-8B (math):** removing imitation learning gives 76.8/67.9/45.0; removing expansion gives 78.2/67.9/44.9.

**Other experiments:** few-shot ARC on Llama-3.2-1B (ICL 0, TTT 10, SEAL 72.5, Sleep 80). Class-incremental learning (CLINC, Banking77, DBpedia), long context (MK-NIAH, LongHealth, QASPER), Manchu/Kalamang translation, and BABILong to 10M tokens are reported **as figures only**, with no numbers in the text.

**Hyperparameters** *[from a summary of Appendix B; not checked against the table]*: lr 5e-6, effective batch 32, 100 Sleep steps, LoRA r=64, α=128. Reaching the same score takes about 3.6–4.8× the wall-clock time of SFT.

---

## 8. Gaps you cannot fill from the paper

1. **No code.** Every part has to be rebuilt from its ancestor repos (§9).
2. **Divergence $\mathcal F$, the RL estimator for LTI, the reward model, $z_0$, $\lambda,\alpha,\gamma$, $d_{\text{low}}$, expert counts, $k,b,m$:** none of these values are given.
3. **How CMS levels map onto a pretrained Qwen3 or Llama:** not stated.
4. **Who the teacher is:** inconsistent (§4.6). **Sign of $\mathcal L_{\text{KS}}$:** inconsistent (§4.5).
5. **How the dream-selection score is computed:** underspecified.
6. **SQuAD base model and grader:** not named. The setup follows SEAL, which used Qwen2.5-7B and a GPT-4.1 judge.

---

## 9. Code map: what to reuse for each stage

| Stage | Reuse | Notes |
|---|---|---|
| Dream generation + LoRA inner loop + ReST-EM | [Continual-Intelligence/SEAL](https://github.com/Continual-Intelligence/SEAL) (`knowledge-incorporation/`) | Closest to Dream-Lite. Has the SQuAD pipeline and the GPT-4.1 grader |
| On-policy distillation (GKD) | TRL [`GKDTrainer`](https://huggingface.co/docs/trl/en/gkd_trainer) | `lmbda`, `beta`, `seq_kd` |
| Gradient-based dream selection | [GREATS](https://github.com/Jiachen-T-Wang/GREATS), [G-DIG](https://github.com/xypan0/G-DIG) | Or, cheaper: cosine similarity between LoRA gradients and a held-out gradient |
| Self-study / context distillation | [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges) | Synthetic conversations about a document plus a distillation loss |
| Expandable low-rank experts | PEFT LoRA (add a new adapter per sleep, freeze the old ones) | *[inferred]* substitute for masked MoE experts |
| Titans / Hope / CMS | [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch); community Hope repos (§0) | Unofficial; not needed for Dream-Lite |

---

## 10. How this maps to Dream-Lite

| Dream-Lite setup | Closest SLEEP / SEAL component |
|---|---|
| (a) passage in prompt | Upper bound (ICL) |
| (b) fine-tune on passage | "Fine-tuned, no dreaming" row (33.4) |
| (c) fine-tune on self-generated QA dreams | Stage 2a + 2c with no selection and no reward. Corresponds to SEAL's "passage + synthetic data, no RL" baseline |
| (d) (c) + replay of old data | Not in SLEEP. SLEEP fights forgetting with expansion plus pruning instead of replay. **This is Dream-Lite's distinct axis** |
| left out: growth, RL | Stage 1 (expansion + KS) and Stage 2d (ReST-EM) |

**Takeaways for the design**
- On SQuAD, **dreaming does most of the work**: 35.7 without dreaming vs 48.9 with it. Consolidation alone adds only ~4 points over the base model, so "dream-lite without growth" keeps the part that matters most for this task.
- The gap between SLEEP/SEAL (~47–49) and a plain dreams baseline mostly comes from **reward filtering (2d)**. Without RL, expect setup (c) to land between rows (b) and SEAL.
- Cheap upgrades within the Dream-Lite scope: gradient-based selection (+1.8 in the ablation) and the random-expert trick (+0.9; it needs MoE, so it doesn't apply to dense Qwen3-1.7B).
- SLEEP's "CPT" does not measure forgetting on an old benchmark. Measuring the before/after drop on GSM8K would be new.

---

## 11. Suggested reading order

1. SEAL ([2506.10943](https://arxiv.org/abs/2506.10943)): the dreaming loop and the SQuAD protocol.
2. GKD ([2306.13649](https://arxiv.org/abs/2306.13649)): the distillation loss.
3. Nested Learning ([2512.24695](https://arxiv.org/abs/2512.24695)), Sec. 3 and 7: frequencies and CMS.
4. Language Models Need Sleep ([2606.03979](https://arxiv.org/abs/2606.03979)), Sec. 3: how the pieces fit together.
5. ReST-EM ([2312.06585](https://arxiv.org/abs/2312.06585)) and GREATS: the reward filtering and selection details.
