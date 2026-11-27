---
title: "Mixture of Experts does not work the way its name says"
subtitle: "What carries the gain, what the experts become, why training ends up there, and what to do about it"
summary: "A sparse MoE layer was designed as a team of specialists with a learned dispatcher. Measured on real models, the specialists are not subject specialists, and in several published tests a fixed hash of the token does as well as the dispatcher. An MoE layer is one very wide feed-forward layer with a block mask, its router is trained by a policy gradient restricted to the experts it already picked, and eight things to try follow for pre-training and for RL."
authors:
  - admin
tags:
  - Mixture of Experts
  - Large Language Models
  - Training Dynamics
  - Sparsity
  - Interpretability
  - Reinforcement Learning
categories:
  - Research
  - Theory
date: '2026-11-27T00:00:00Z'
lastmod: '2026-11-27T00:00:00Z'
math: true
draft: false

citation: |
  @online{li-2026-how-moe-actually-works,
    title  = {Mixture of Experts does not work the way its name says},
    author = {Li, Yingru},
    year   = {2026},
    month  = nov,
    url    = {https://richardli.xyz/post/how-moe-actually-works/}
  }

sidebar:
  toc: true
---

{{% callout note %}}
1. The design story predicts subject experts, context-driven routing, and a router that has to be learned well. None of the three holds up cleanly in measurements.
2. An MoE layer is exactly one very wide feed-forward layer with a block mask. Read that way, the gain comes from storing *M* blocks while computing *K*.
3. Specialization is real, at the level of small operations, with many distinct ones in each expert. It looks like a by-product of sparsity more than the plan behind it.
4. The router is trained by a policy gradient restricted to the experts it already picked. It gets no signal about the others, so experts it ignores stay untrained, and without outside pressure some of them die.
5. Eight things to try follow, six for pre-training and two for RL. The cheapest: center the router's input, count tokens per expert, and replay the sampler's routing.
{{% /callout %}}

A sparse MoE layer was designed as a team of specialists with a learned dispatcher. Measured on real models, the specialists are not subject specialists, and in several published tests a fixed hash of the token does as well as the dispatcher. This post works out what carries the gain instead, what the experts turn into, why training ends up there, and what to do about it in pre-training and in RL.

Each claim carries its standing: <span class="claim-tag claim-tag--m">measured</span> reported in a paper, <span class="claim-tag claim-tag--d">derived</span> worked out here and checked by code, <span class="claim-tag claim-tag--t">toy</span> a small MoE trained for this post, <span class="claim-tag claim-tag--i">interpretation</span> a reading that fits the evidence. The exact numbers, the finer points and the numerical checks sit in folded blocks and in the appendix, so the main text can be read straight through.

## 1. The layer, and the story behind it

A Transformer block applies attention and then a feed-forward network (FFN) to each token's vector. A sparse MoE layer replaces that one FFN with {{< math >}}$M${{< /math >}} of them, the experts, and adds a router that picks {{< math >}}$K${{< /math >}} of them for each token.

{{< math >}}
$$
h = W_r\,x,
$$
$$
S = \text{top-}K(h),
$$
$$
g_i = \frac{e^{h_i}}{\sum_{j\in S} e^{h_j}},
$$
$$
y = \sum_{i\in S} g_i\,E_i(x)
$$
{{< /math >}}

Here {{< math >}}$x${{< /math >}} is the token's vector, {{< math >}}$h${{< /math >}} holds one router score per expert, {{< math >}}$S${{< /math >}} is the set of the {{< math >}}$K${{< /math >}} highest scores, the gates {{< math >}}$g_i${{< /math >}} are a softmax over those scores, and {{< math >}}$y${{< /math >}} is the layer's output. Only the experts in {{< math >}}$S${{< /math >}} run.

{{< livefig "moe-forward" `**Figure 1. One token through an MoE layer: every expert is scored, a few run.** Scoring is one matrix multiply, so it is cheap to score all <i>M</i> experts. Running them is the expensive part, and only <i>K</i> run. The logits here are random numbers seeded by the token, for illustration.` 520 `One token through an MoE layer: every expert is scored, a few run` >}}

Two numbers describe the layer. {{< math >}}$M${{< /math >}} sets how many parameters are stored. {{< math >}}$K${{< /math >}} sets how much is computed for a token. <span class="claim-tag claim-tag--m">measured</span> [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) stores 8 experts per layer and runs 2, so a token touches 28% of the parameters. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) runs 8 of 256 routed experts per layer, and a token touches 5.5%.

The idea is from 1991. Jacobs, Jordan, Nowlan and Hinton let a gate split the input space into regions and let each expert become good on its own region. [Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538) made the gate sparse so that only {{< math >}}$K${{< /math >}} experts run. The name carries a strong picture: the experts know different things, and the router learns who knows what. That picture makes three predictions that can be tested.

## 2. Three predictions, and what is measured

### Prediction 1: experts specialize by subject

<span class="claim-tag claim-tag--m">measured</span> The [Mixtral](https://arxiv.org/abs/2401.04088) authors counted which experts fire on eight subject subsets of The Pile, from ArXiv to PubMed abstracts. The distributions are nearly the same. Routing follows syntax instead: Python indentation tokens go to the same experts, and neighboring tokens share an expert about twice as often as chance. [OpenMoE](https://arxiv.org/abs/2402.01739) and a 2026 replication on gpt-oss-20b found the same flat picture.

### Prediction 2: the route depends on what a token means in context

<span class="claim-tag claim-tag--m">measured</span> In [OpenMoE](https://arxiv.org/abs/2402.01739), a token such as “ed”, “an” or a comma goes to the same few experts whatever surrounds it, and the assignment looked the same from the first saved checkpoint to the last. Later work finds that context does count for something. Experts overlap a little more when a word keeps its meaning, and in one 2026 model a probe on learned features predicts the route far better than the token alone. Token identity explains a lot of routing, most of all in early open models. Context adds a measurable share in newer ones.

### Prediction 3: the router has to learn a good assignment

<span class="claim-tag claim-tag--m">measured</span> [Hash Layers](https://arxiv.org/abs/2106.04426) (Roller et al., NeurIPS 2021) removed the router. The expert is chosen by a fixed hash of the token ID. With one sparse layer, the hash matched or beat Switch Transformer's learned router in most tests, and its lead grew with the number of experts. The learned router won with five sparse layers and on one large-vocabulary test. The largest model in the paper has 4.5B parameters.

One row in that paper says more than the headline. Putting similar tokens in the same expert, as the design story recommends, is worse than a random hash. Spreading each cluster of similar tokens across all the experts recovers the loss. The authors' hypothesis is that the fine distinctions a model has to make are between similar tokens, so similar tokens should get different parameters. <span class="claim-tag claim-tag--t">toy</span> The same ordering appears in a small MoE trained for this post (Section 5): routing by cluster loses to a random hash.

{{< livefig "moe-routes" `**Figure 2. The same three passages, routed by subject and routed by token identity.** Tokens are routed one at a time and colored by their first-choice expert out of eight. Bars give each expert's share of the passage so far. Under subject routing every passage belongs to one expert. Under token routing no passage belongs to one expert, and “the” has one color everywhere. That absence of subject structure is what Mixtral and OpenMoE measured. The second rule is literally a Hash Layer. This figure illustrates the two rules. It is not data from a model.` 560 `The same three passages, routed by subject and routed by token identity` >}}

**Table 1. The design story against the measurements.**

| The design story predicts | What was measured | Outcome |
|---|---|---|
| Experts divide the data by subject | Expert usage is nearly identical across subject corpora in three studies | Not found |
| The route depends on meaning in context | Token identity fixes most of the route. Context has a small effect in most models tested and a larger one in one 2026 model | Partly |
| A learned router is needed | A fixed hash matches or beats it in most single-layer tests up to 1.28B parameters, and loses in two settings. Not tested at frontier scale | Often not, at tested scale |

<details class="spoiler">
<summary>The numbers behind the three predictions</summary>

**Prediction 1.** <span class="claim-tag claim-tag--m">measured</span> The [Mixtral](https://arxiv.org/abs/2401.04088) authors counted which experts fire on eight subsets of The Pile (ArXiv, GitHub, PubMed abstracts, philosophy papers and others) at layers 0, 15 and 31. The expert distributions are nearly the same across subsets. Only the synthetic DM Mathematics set differs slightly. Routing tracks syntax instead: Python indentation tokens go to the same experts, and at layer 15 consecutive tokens share a first-choice expert 23.6–28.4% of the time against 12.5% by chance.

[OpenMoE](https://arxiv.org/abs/2402.01739) found the same flat picture across its pre-training domains and across programming languages, though it did see specialization by natural language and by token position. [RouterInterp](https://arxiv.org/abs/2610.11775) repeated the Pile test on gpt-oss-20b in 2026. Routing stayed close to corpus-proportional, with a few experts spiking on GitHub or ArXiv. Earlier, [ST-MoE](https://arxiv.org/abs/2202.08906) reported that some encoder experts pick up categories such as punctuation, verbs and proper names, while decoder experts show no meaningful specialization.

**Prediction 2.** <span class="claim-tag claim-tag--m">measured</span> In [OpenMoE](https://arxiv.org/abs/2402.01739), a token such as “ed”, “an” or a comma goes to the same few experts whatever surrounds it. Experts collect lexically similar tokens: “can”, “will” and “would” sit in one, “have”, “has” and “had” in another. The assignment was already in place at the first saved checkpoint, 200B tokens in, and looked the same at 1T, through a change of data mix (code cut from 52.25% to 15% by the paper's data table, 20% by its text) and a change of training objective. The evidence is one figure with two panels, for the tokens “ed” and “an”, and checkpoints exist only every 200B tokens, so the authors' claim that routing is learned early is an inference.

Later work complicates this. [Olson et al.](https://arxiv.org/abs/2502.10928) held a word fixed and changed its sense, across six models from three families. Experts overlap more when the meaning is preserved. The effect is statistically clear, with p < 0.001 in every model, and strongest in the middle layers. Read off the paper's Figure 2, it is small in absolute size: a difference of about 0.02 to 0.05 in chance-normalized overlap. In the RouterInterp paper, a probe on sparse-autoencoder features predicts routing in gpt-oss-20b at layer 12 with macro-F1 0.73, against 0.30 for a unigram baseline. The gap is narrower at some other layers, 0.57 against 0.47 at layer 4. Token identity explains a lot of routing, most of all in early open models. Context adds a measurable share in newer ones.

**Prediction 3.** <span class="claim-tag claim-tag--m">measured</span> [Hash Layers](https://arxiv.org/abs/2106.04426) (Roller et al., NeurIPS 2021) removed the router. The expert is chosen by a fixed hash of the token ID. On Reddit data with 64 experts in one sparse layer (751M parameters), a hash balanced by token frequency reaches 23.16 validation perplexity. Switch Transformer with its learned router reaches 23.65. With 128 experts (1.28B parameters) the gap grows to about 0.6. The result is mixed elsewhere: Switch wins with five sparse layers of 16 experts (23.00 against 23.21), and with one sparse layer on WikiText-103 with its full vocabulary (31.76 against 32.32). The largest model in the paper has 4.5B parameters.

Two rows in that paper say more than the headline. Hashing by *k*-means clusters of token embeddings puts similar tokens in the same expert, as the design story recommends. It is worse than a purely random hash: 23.90 against 23.22. Spreading each cluster across all experts recovers the loss (23.17). The authors' hypothesis, for which they call this partial evidence, is that the fine distinctions a model has to make are between similar tokens, so similar tokens should get different parameters. And when the Switch router is fed the token identity instead of the hidden state, it improves from 23.65 to 23.43.

<span class="claim-tag claim-tag--t">toy</span> The same ordering appears in a small MoE trained for this post (setup in Section 5). Tokens come in clusters of look-alikes, and each token needs its own answer. With one fixed expert per token, routing by cluster ends at loss 0.79, a random hash at 0.45, and a hash that splits every cluster across the experts at 0.39, averaged over 8 seeds. The gap widens as the look-alikes get closer: 1.50 against 0.66 and 0.56. Routing by cluster loses to the random hash in 6 of the 8 seeds, and in all 8 when the look-alikes are closest. The extra gain from splitting clusters on purpose is smaller than the spread between seeds.

</details>

## 3. What carries the gain

### One wide layer with a block mask

<span class="claim-tag claim-tag--d">derived</span> Write each expert as a two-layer MLP, {{< math >}}$E_i(x) = V_i \sigma (U_i x)${{< /math >}}. Stack the input matrices into one tall matrix {{< math >}}$U${{< /math >}} and place the output matrices side by side in {{< math >}}$V${{< /math >}}. Then the whole layer is

{{< math >}}
$$
y \;=\; V\,\big(m(x)\odot\sigma(U x)\big)
$$
{{< /math >}}

where the mask {{< math >}}$m(x)${{< /math >}} holds {{< math >}}$g_i${{< /math >}} on every hidden unit of block {{< math >}}$i${{< /math >}} when expert {{< math >}}$i${{< /math >}} is selected, and 0 on all other blocks. This is an identity. An MoE layer is a single FFN whose hidden layer is {{< math >}}$M${{< /math >}} times wider, multiplied by a mask that zeroes all but {{< math >}}$K${{< /math >}} blocks. The zeros are known before the blocks are computed, and that is the whole saving.

The FFN is where a Transformer keeps most of its weights, about two-thirds of them in a typical model. [Geva et al.](https://arxiv.org/abs/2012.14913) showed that it behaves as a key–value memory: each hidden unit's input weights detect a pattern and its output weights promote the tokens likely to follow it. <span class="claim-tag claim-tag--i">interpretation</span> Read that way, the router is a coarse address that says which blocks to open, and the activation function is the fine address that says which units inside respond.

{{< livefig "moe-wide" `**Figure 3. An MoE layer drawn as one wide hidden layer. Cutting the blocks smaller keeps the cost and multiplies the addresses.** A new token arrives every two seconds and opens a different set of blocks. All three settings store 64 hidden units and compute 16. What changes is how many distinct sets of units a token can be given. This count is the argument DeepSeekMoE makes for fine-grained experts.` 400 `An MoE layer drawn as one wide hidden layer. Cutting the blocks smaller keeps the cost and multiplies the addresses` >}}

<span class="claim-tag claim-tag--i">interpretation</span> Three things follow from the wide-layer view, and each matches something measured. The results are from the papers. Connecting them to this view is a reading.

- **Capacity grows faster than cost.** In the [scaling-law fit](https://arxiv.org/abs/2402.07871) of Krajewski et al., a compute-optimal MoE at {{< math >}}$10^{20}${{< /math >}} FLOPs matches a dense model trained with about 20 times more compute, and the gap widens with scale.
- **Any stable, balanced address works to first order.** A hash is a stable, balanced address.
- **A finer address is worth more than a smarter one.** [DeepSeekMoE](https://arxiv.org/abs/2401.06066) cuts each expert into smaller ones, opens proportionally more, and adds shared experts that are always on. The scaling-law study found that experts as wide as the original FFN are almost never the best choice. Hash Layers made the same move in 2021, by splitting each expert into segments chosen by independent hashes.

### Where the theory fits

<span class="claim-tag claim-tag--i">interpretation</span> The main theory result is [Chen et al. (NeurIPS 2022)](https://arxiv.org/abs/2208.02813). On a classification problem built from clusters that each need a different rule, a single two-layer network provably cannot exceed 87.5% accuracy, while an MoE with nonlinear experts succeeds and its router provably learns the cluster centers. That confirms the design story for data where inputs that look alike need the same function. The clustering row in Hash Layers says language is different in one respect: tokens that look alike need different handling. Both results fit one statement. A good partition makes each expert's own sub-problem easy, and whether that means grouping similar inputs or separating them depends on the task.

## 4. What the experts become

<span class="claim-tag claim-tag--m">measured</span> Experts are not subject specialists, and they are not interchangeable either. Four results from 2025 and 2026 say what they are.

- **Operations, not subjects.** [Herbst et al. (ICML 2026)](https://arxiv.org/abs/2604.02178) labeled hundreds of experts in three models. Experts are neither subject specialists nor plain token processors. They carry out fine-grained operations. An expert may be confined to LaTeX, but its job is closing delimiters.
- **Cleaner units, and cleaner as routing gets sparser.** In the same study, neurons inside experts are less polysemantic than neurons in dense FFNs, and more so in models whose routing is sparser.
- **Many distinct jobs per expert.** [RouterInterp](https://arxiv.org/abs/2610.11775) explains each expert's routing with sparse-autoencoder features. An expert's top 20 features fall into about 11 distinct clusters on average. A subject expert would give one.
- **A core and a periphery.** [Wang et al. (2026)](https://arxiv.org/abs/2601.03425) find that a few experts per layer, a “standing committee”, take a large share of the routing weight whatever the subject. The committee takes question words and common function words. Peripheral experts are called in for chemical symbols, biomedical identifiers and financial terms.

{{< livefig "moe-found" `**Figure 4. Experts, named by what they were found to do.** Five experts from two papers (Herbst et al. 2026 and OpenMoE), next to the kind of expert the name suggests. The examples under each card are the ones the papers give. For the four experts from Herbst et al. that is the highlighted trigger, then the tokens the expert promotes. For the OpenMoE expert it is three of the tokens routed to it.` 420 `Experts, named by what they were found to do` >}}

> <span class="claim-tag claim-tag--i">interpretation</span> A block that is open for only a slice of the tokens has fewer things to represent at once, so its features interfere less. For that to help, the router does not have to group related things. The slice only has to be stable. This one mechanism accounts for four of the findings so far: a random hash works, clustering similar tokens hurts, finer blocks help, and an expert's contents look like a bag of many small jobs. Specialization is what sparsity produces. It is not what the router was aiming at.

<details class="spoiler">
<summary>How the four results were obtained, with their numbers</summary>

- **Operations, not subjects.** [Herbst et al. (ICML 2026)](https://arxiv.org/abs/2604.02178) probed twelve models, six of them MoE, labeled hundreds of experts automatically in three, scored every label with a second model, and checked a sample of 30 by attribution. Experts are neither subject specialists nor plain token processors. They carry out fine-grained operations. An expert may be confined to LaTeX, but its job is closing delimiters.
- **Cleaner units, and cleaner as routing gets sparser.** In the same study, neurons inside experts are less polysemantic than neurons in dense FFNs. Mixtral, where a quarter of the experts are active, scores noticeably lower than Qwen3-30B-A3B, where about 6% are. OLMoE with 1B active parameters beats the dense OLMo-7B, which points to sparsity and not size. A [toy-model study](https://arxiv.org/abs/2510.23671) finds the same direction.
- **Many distinct jobs per expert.** [RouterInterp](https://arxiv.org/abs/2610.11775) explains each expert's routing with sparse-autoencoder features. An expert's top 20 features fall into about 11 semantically distinct clusters on average, where random assignment would give about 17. A subject expert would give one.
- **A core and a periphery.** [Wang et al. (2026)](https://arxiv.org/abs/2601.03425) find that a few experts per layer, a “standing committee” of one to five, take a large share of the routing weight across nine groups of MMLU subjects: up to 70.5% in DeepSeek-V2-Lite, up to 67.0% in Qwen3-30B-A3B, and 14.5% to 44.0% across the layers of OLMoE. The committee takes question words, question marks and common function words. Peripheral experts are called in for chemical symbols, biomedical identifiers and financial terms.

</details>

## 5. Why training ends up there

### The router's gradient

<span class="claim-tag claim-tag--d">derived</span> Differentiate the loss {{< math >}}$L${{< /math >}} through the layer of Section 1 with respect to one router logit:

{{< math >}}
$$
\frac{\partial L}{\partial h_k} \;=\; g_k\,\big\langle \nabla_y L,\; E_k(x) - y \big\rangle \quad \text{for } k\in S,
$$
$$
\frac{\partial L}{\partial h_k} = 0 \quad \text{for } k\notin S.
$$
{{< /math >}}

Define {{< math >}}$r_k = -\langle \nabla_y L, E_k(x)\rangle${{< /math >}}, the first-order drop in loss per unit of expert {{< math >}}$k${{< /math >}}'s output added to {{< math >}}$y${{< /math >}}. A gradient step of size {{< math >}}$\eta${{< /math >}} taken on the logits changes them by

{{< math >}}
$$
\Delta h_k \;=\; \eta\, g_k\,\big(r_k - \bar r\big),
$$
$$
\bar r = \sum_{j\in S} g_j\, r_j .
$$
{{< /math >}}

This has the form of the policy gradient for a softmax policy. The actions are experts, the reward is {{< math >}}$r_k${{< /math >}}, and the baseline {{< math >}}$\bar r${{< /math >}} is the reward of the current mixture. It carries no sampling noise, because the layer averages over the selected experts instead of sampling one. It is also truncated to the set {{< math >}}$S${{< /math >}}. Four consequences follow.

- **No counterfactual.** An expert outside the top-{{< math >}}$K${{< /math >}} sends nothing back for that token, however good its output would have been.
- **The update is zero-sum inside {{< math >}}$S${{< /math >}}.** The changes add to zero, {{< math >}}$\sum_{k\in S} \Delta h_k = 0${{< /math >}}. A token cannot lift its selected experts above the rest as a group. One rises, another sinks, and the selected expert with the lowest reward drifts down toward the outsiders.
- **Outsiders get their turn, exponentially slowly.** An outsider is tried only after a selected expert has sunk below it. If the outsider sits a logit gap {{< math >}}$\Delta${{< /math >}} below, that wait grows as {{< math >}}$e^{2\Delta}${{< /math >}}. The size of the router's logits is therefore an exploration setting.
- **With one expert and a renormalized gate there is no gradient.** If {{< math >}}$K = 1${{< /math >}} then {{< math >}}$g = 1${{< /math >}} and {{< math >}}$y = E_k(x)${{< /math >}}, so the bracket is zero. [Switch Transformer](https://arxiv.org/abs/2101.03961) trains with {{< math >}}$K = 1${{< /math >}} by gating with the probability {{< math >}}$p_i${{< /math >}} taken over all {{< math >}}$M${{< /math >}} experts. Its update reinforces the chosen expert whenever its output helps at all, with no baseline and still no counterfactual.

<span class="claim-tag claim-tag--m">measured</span> The published fixes add exploration or estimate the missing counterfactual. Shazeer et al. add noise to the logits, and the Switch paper calls the problem by its name, an exploration–exploitation dilemma. [Default MoE](https://arxiv.org/abs/2504.12463) fills in each unselected expert's output with a running average of its past outputs, so every logit gets a signal.

<details class="spoiler">
<summary>The waiting time, the Switch update, and Default MoE's results</summary>

- **Outsiders get their turn, exponentially slowly.** Take {{< math >}}$K = 2${{< /math >}}: selected experts {{< math >}}$a${{< /math >}} and {{< math >}}$b${{< /math >}} with frozen rewards {{< math >}}$r_a > r_b${{< /math >}} and logit gap {{< math >}}$d_0 = h_a - h_b${{< /math >}}, and an outsider sitting {{< math >}}$\Delta${{< /math >}} below {{< math >}}$b${{< /math >}}. In continuous time, {{< math >}}$b${{< /math >}} sinks far enough to be replaced after

  {{< math >}}
  $$
  t(\Delta) \;=\; \frac{F(d_0+2\Delta) - F(d_0)}{2\,\eta\,(r_a - r_b)},
  $$
  $$
  F(d) = e^{d} + 2d - e^{-d}
  $$
  {{< /math >}}

  steps, which grows as {{< math >}}$e^{2\Delta}${{< /math >}}. The size of the router's logits is therefore an exploration setting.
- **With one expert and a renormalized gate there is no gradient.** If {{< math >}}$K = 1${{< /math >}} then {{< math >}}$g = 1${{< /math >}} and {{< math >}}$y = E_k(x)${{< /math >}}, so the bracket is zero. Shazeer et al. note that the gates have non-zero derivatives when more than one expert is chosen. [Switch Transformer](https://arxiv.org/abs/2101.03961) trains with {{< math >}}$K = 1${{< /math >}} by gating with the probability {{< math >}}$p_i${{< /math >}} taken over all {{< math >}}$M${{< /math >}} experts, {{< math >}}$y = p_i E_i(x)${{< /math >}}. The update becomes {{< math >}}$\Delta h_i = \eta p_i(1 - p_i) r_i${{< /math >}}: the chosen expert is reinforced whenever its output helps at all. The baseline is zero, there is still no counterfactual, and while its output helps, nothing pushes the chosen expert back down. The other logits move only through the normalization and learn nothing about their own experts.
- **The fixes are exploration or estimated counterfactuals.** <span class="claim-tag claim-tag--m">measured</span> Shazeer et al. add noise to the logits. The Switch paper compares argmax, sampling, input dropout and input jitter in an appendix and calls the problem by its name, an exploration–exploitation dilemma. [Default MoE](https://arxiv.org/abs/2504.12463) fills in each unselected expert's output with a running average of its past outputs, so every logit gets a signal. At 2B parameters with 8 experts and one selected, that raised the average of 13 benchmarks from 46.9 to 47.9 and reached the target perplexity with about 9% fewer tokens. With 32 experts and 4 selected, the average moved from 49.4 to 49.5.

</details>

{{< livefig "moe-grad" `**Figure 5. The derived update, iterated. The router reaches a better expert only by pushing losers down.** The expert outputs and the loss direction are frozen, so only the router moves, for 600 steps of the update above. Left: the outputs of five experts for one token, the direction that lowers the loss, and the mixture <i>y</i>. Right, top: each expert's reward <i>r<sub>k</sub></i>. Right, bottom: its router logit. Filled bars are selected. E4 is the best expert and starts fourth. At <i>K</i> = 2, with the logits as drawn, it is first tried at step 46 and takes over. With logits twice as large it is first tried at step 310, and three times as large at step 2,753, long after this animation ends. In real training the selected experts also fit the token, the loss gradient shrinks, and the signal fades before the slow demotion finishes.` 470 `The derived update, iterated. The router reaches a better expert only by pushing losers down` >}}

### The feedback loop

A greedy learner with this kind of feedback has a known failure. Whatever is picked early gets the data, gets better, and gets picked again. <span class="claim-tag claim-tag--m">measured</span> Shazeer et al. reported it in 2017: the gate converges to giving large weights to “the same few experts”, and the imbalance feeds itself because the favored experts train faster. Every recipe cited here adds pressure against it.

{{< livefig "moe-loop" `**Figure 6. The feedback loop.** The table below says where each training trick cuts into it.` 260 `The feedback loop, and where each training trick cuts into it` >}}

| Trick | What it changes in the loop | Used in |
|---|---|---|
| Noise on logits, input jitter, input dropout | Lets experts outside the top-{{< math >}}$K${{< /math >}} be selected sometimes | Shazeer 2017, Switch |
| Default vector for unselected experts | Gives every router logit a signal from an estimate of the missing output | Default MoE |
| Auxiliary balance loss | Pushes average router probabilities toward uniform, through the model's gradients | Shazeer 2017, Switch |
| Capacity limit per expert | Tokens over an expert's quota are not processed by it | Switch, OpenMoE |
| Load bias on selection only | Moves tokens off overloaded experts without touching the model's gradients | DeepSeek-V3 |

### A tiny MoE, trained for real

<span class="claim-tag claim-tag--t">toy</span> To see the loop under real gradients, this post trains a small MoE layer: 96 token types with Zipf frequencies, 8 experts of width 8, 2 selected per token, Adam, 4,000 steps, 8 seeds. Each token vector is a shared component plus a cluster center plus a token-specific part, as in a residual stream, and each token has its own target.

- **Experts die without balance pressure.** With no balancing, 1.6 of the 8 experts end with no tokens and the loss is 0.44. With Switch's auxiliary loss none die and the loss is 0.21.
- **The shared component of the token vectors is what kills them.** Without it almost no expert dies. With it doubled, more than three of the eight do. <span class="claim-tag claim-tag--d">derived</span> The router is linear, so a plain gradient step on token {{< math >}}$x${{< /math >}} changes expert {{< math >}}$k${{< /math >}}'s logit on any other token {{< math >}}$x'${{< /math >}} in proportion to {{< math >}}$\langle x, x'\rangle${{< /math >}}. A shared component adds the same positive term to every pair's inner product, so each win raises the expert on every token.
- **Centering the router's input fixes it without a balance term.** Subtract the mean token vector before routing: no dead experts and loss 0.15, the best of the six combinations of balancing (none, auxiliary loss, load bias) and centering.

{{< livefig "moe-toy" `**Figure 7. One training run of the tiny MoE, three ways: how the load spreads over eight experts.** The same data and seed under three settings, 3,000 steps of Adam. Each bar is an expert's share of the token slots; the line marks the fair share of one eighth. Without balancing, experts 4 and 8 lose their tokens within the first 50 steps and never get them back. Expert 3 is also empty until step 1,750 and then recovers. This is one seed, and in it the auxiliary loss ends lower than centering. Over eight seeds and 4,000 steps the order is the reverse. The averages are in the text.` 330 `One training run of the tiny MoE, three ways: how the load spreads over eight experts` >}}

Four more results are in the fold below. An expert the router has ignored looks bad when it is finally tried. Routing kept drifting through training and did not freeze early. A change of data mix unbalanced the load without reorganizing the routing. And no update rate made the load bias both effective and stable.

<details class="spoiler">
<summary>Four more toy results, with the numbers</summary>

- **The shared component of the token vectors is what kills them.** In a separate sweep of 3,000 steps, 0.125 experts die with the shared component removed, 1.75 at its default size, and 3.375 at double. <span class="claim-tag claim-tag--d">derived</span> The router is linear, so a plain gradient step on token {{< math >}}$x${{< /math >}} changes expert {{< math >}}$k${{< /math >}}'s logit on any other token {{< math >}}$x'${{< /math >}} in proportion to {{< math >}}$\langle x, x'\rangle${{< /math >}}. A shared component adds the same positive term to every pair's inner product, so each win raises the expert on every token.
- **An untrained expert looks bad.** At the end of the 3,000-step runs at the default size, swapping a token's weakest selected expert for the best outsider lowers the loss for 6% of the token mass. The median swap multiplies the loss by 2.7. OpenMoE inferred, without testing it, that this is why routing stays fixed: a token sent to an expert that has not seen it raises the loss.
- **Routing did not freeze early here.** A tenth of the way in, a third of the token mass had its final pair of experts. Halfway, two-thirds. Routing kept drifting while the loss was falling, ten times more slowly by the end. Whatever happened in OpenMoE's model, early lock-in does not follow from the mechanism.
- **A change of data mix unbalances the load and leaves most of the routing alone.** Permuting the token frequencies at step 2,000 of 4,000 about triples the loss at that step. The effective number of experts in use, the inverse of the sum of squared load shares, drops from 7.2 to 6.4 with a centered router input. By the end the loss is back within noise of the unchanged run, 0.16 against 0.15. The routing does not reorganize: 28% of the token types changed experts in the second half, against 26% with no change of mix. With the auxiliary loss it is 40% against 26%. The balance pressure moves tokens. The new data alone does not. OpenMoE saw the assignment survive a change of data mix at scale.
- **No update rate made the load bias both effective and settled.** Six rates from {{< math >}}$10^{-4}${{< /math >}} to {{< math >}}$2 \times 10^{-2}${{< /math >}} were tried. The paper that introduced the bias uses {{< math >}}$10^{-3}${{< /math >}}, and reports {{< math >}}$10^{-4}${{< /math >}} as too slow and {{< math >}}$10^{-2}${{< /math >}} as fluctuating. The two smallest did little: loss 0.41 and 0.39, close to the 0.44 of no balancing. Every rate from {{< math >}}$10^{-3}${{< /math >}} up kept all experts alive and kept moving tokens. In the last quarter of training, 17 to 20% of the token mass changed experts between evaluations 50 steps apart, against 10% with the auxiliary loss and 3% with no balancing. The best loss was 0.25, against 0.21 for the auxiliary loss. The bias moves by the sign of the load error in one batch of 128 tokens, which is a noisy estimate. Large batches make it far less noisy, so this result is about the toy.

</details>

Whether uniform load is the right target is open. The standing-committee study argues that balance losses may be fighting a natural split into a general core and a specialized periphery. The authors did not test that claim, and their masking experiment has no random-expert control. DeepSeekMoE's shared experts are an architectural version of the same idea: the core is made explicit and always on.

## 6. What sparsity costs

How many experts should run for a token? [Zhu et al. (ICML 2026)](https://openreview.net/forum?id=dNLXIVeCqn) give a convergence bound for SGD on a deep MoE. Read at face value, it says that more experts per token converge faster per step, and that a fixed compute budget has a best {{< math >}}$K${{< /math >}}. [Appendix A](#appendix-a-the-convergence-bound-for-k-read-closely) reads the bound closely. In short, its dependence on {{< math >}}$K${{< /math >}} comes from a loose step in the proof, its best {{< math >}}$K${{< /math >}} is set by a noise term that nothing measures, and at {{< math >}}$K = 1${{< /math >}} the paper's gate gives the router no gradient. <span class="claim-tag claim-tag--t">toy</span> In the tiny MoE the best setting at equal compute is {{< math >}}$K = 2${{< /math >}}, as in the paper's language runs.

One cost of sparsity does hold up. <span class="claim-tag claim-tag--d">derived</span> The gradient of the batch loss for expert {{< math >}}$i${{< /math >}} is a sum over the tokens routed to it. With batch size {{< math >}}$B${{< /math >}} and token share {{< math >}}$\ell_i${{< /math >}}, its signal-to-noise ratio is

{{< math >}}
$$
\mathrm{SNR}_i \;\approx\; B\,\ell_i\;\frac{\Vert\bar q_i\Vert^2}{\sigma_i^2} \quad \text{for small } \ell_i ,
$$
{{< /math >}}

where {{< math >}}$\bar q_i${{< /math >}} is the mean per-token gradient over the expert's tokens and {{< math >}}$\sigma_i^2${{< /math >}} their mean squared norm. An expert in a layer with {{< math >}}$M/K = 32${{< /math >}} learns from a batch 32 times smaller than the attention weights next to it. This is the real cost of sparsity that the bound was reaching for.

## 7. What to do with this

Each item says where it comes from. None has been tested at scale here.

### In pre-training

1. **Center the router's input.** <span class="claim-tag claim-tag--t">toy</span> Route on {{< math >}}$x - \mu${{< /math >}}, with {{< math >}}$\mu${{< /math >}} a running mean of the layer's input. The shared component of the residual stream ties every token's logits together, and that coupling is what let experts die in the toy. One line of code, and the cheapest ablation on this list.
2. **Count tokens per expert, not per batch.** <span class="claim-tag claim-tag--d">derived</span> An expert's gradient has signal-to-noise proportional to {{< math >}}$B \cdot K/M${{< /math >}}. When you raise {{< math >}}$M/K${{< /math >}}, raise the batch with it or lower the experts' learning rate.
3. **Treat the size of router logits as an exploration setting.** <span class="claim-tag claim-tag--d">derived</span> The wait before an outsider is tried grows as {{< math >}}$e^{2\Delta}${{< /math >}} in the logit gap. A small router initialization, a temperature, or a penalty on logit size such as ST-MoE's router z-loss keeps that wait short early in training. Log the margin between the {{< math >}}$K${{< /math >}}-th and {{< math >}}$(K+1)${{< /math >}}-th logit.
4. **Choose the gate's normalization on purpose.** <span class="claim-tag claim-tag--d">derived</span> A softmax over the selected set is zero-sum and cannot train a router at {{< math >}}$K = 1${{< /math >}}. A gate that is not renormalized over the selected set, as in Switch, entrenches whatever was chosen. When two values of {{< math >}}$K${{< /math >}} are compared, check that the router receives a gradient at both.
5. **Give the router the token.** <span class="claim-tag claim-tag--m">measured</span> In Hash Layers, the Switch router improves when it is fed the token identity, and putting similar tokens in one expert hurts. Add a token-identity term to the router logits, or start routing from a balanced hash, and let the hidden state adjust it.
6. **Log the softmax mass outside the top-{{< math >}}$K${{< /math >}}.** <span class="claim-tag claim-tag--d">derived</span> For one layer, sparse and dense routing differ in output by at most twice that mass times the largest expert output norm, a step in the proof of Zhu et al.'s first theorem. It measures how far training is from dense, and it gives a per-token rule for {{< math >}}$K${{< /math >}}: take experts until the remaining mass is under a budget.

### In RL on an MoE

7. **Make routing part of the recorded trajectory.** <span class="claim-tag claim-tag--m">measured</span> A token whose {{< math >}}$K${{< /math >}}-th and {{< math >}}$(K+1)${{< /math >}}-th logits are nearly tied flips experts under any numerical difference between sampler and trainer, and the output jumps however small the difference was. [Ma et al.](https://arxiv.org/abs/2510.11370) measured this on Qwen3-30B-A3B: about 10% of the routers select different experts in the trainer than in the sampler. Replaying the sampler's top-{{< math >}}$K${{< /math >}} mask in the trainer, with the gates still computed from the trainer's logits, stopped the RL collapses. Log the flip rate and the tie margin either way.
8. **Count tokens per expert in the RL batch.** <span class="claim-tag claim-tag--i">interpretation</span> RL batches are small and far from the pre-training mix. By item 2, experts hit by a handful of tokens get high-variance updates, and OpenMoE showed that load goes out of balance on inputs unlike the pre-training data. Log per-expert token counts per update, and consider accumulating over more tokens before updating rarely hit experts.

### Open questions

- Does centering the router input remove the need for balance pressure at scale, or only in a toy?
- Does a learned router beat a hash at frontier scale, or does the hash's lead keep growing as experts multiply?
- What makes routing sensitive to context: model size, depth, starting from a dense model, or the training recipe?
- Is uniform balance suppressing a useful core and periphery?
- Is there a theory in which separating similar inputs is optimal, to set beside the cluster theory of Chen et al.?

## Appendix A: the convergence bound for K, read closely

[Zhu et al. (ICML 2026)](https://openreview.net/forum?id=dNLXIVeCqn) ask how {{< math >}}$K${{< /math >}} changes the convergence of SGD on a deep MoE. <span class="claim-tag claim-tag--m">measured</span> Their bound on the average squared gradient norm after {{< math >}}$T${{< /math >}} steps, in the form of their second corollary, has three parts: the initial gap {{< math >}}$Z_1${{< /math >}}, gradient noise {{< math >}}$Z_2${{< /math >}}, and a routing term {{< math >}}$Z_3${{< /math >}} that comes from tokens switching experts between steps.

{{< math >}}
$$
\frac1T\sum_{t \lt T}\mathbb{E}\,\Vert\nabla F(w^t)\Vert^2 \;\le\; \mathcal{O}\!\left(\Big(Z_1 + Z_3\big(1-\tfrac KM\big)\Big)\,\frac{1+\frac MK}{\sqrt T}\right) + \mathcal{O}\!\left(\frac{Z_2}{\sqrt T}\right)
$$
{{< /math >}}

Read at face value it says two things. More experts per token converge faster per step, by the factor {{< math >}}$1 + M/K${{< /math >}}. And under a fixed compute budget {{< math >}}$R${{< /math >}}, where a token costs {{< math >}}$r_0 = r_s + N r_g + K N r_e${{< /math >}} for the non-MoE parts, {{< math >}}$N${{< /math >}} routers and {{< math >}}$KN${{< /math >}} experts, there is a best {{< math >}}$K${{< /math >}}, because the budget buys {{< math >}}$T = R/r_0${{< /math >}} steps. Bounding {{< math >}}$1 - K/M${{< /math >}} by 1 and dropping constants as the paper does, the bound is proportional to

{{< math >}}
$$
b(K) \;=\; \Big(1 + \frac MK + \rho\Big)\,\sqrt{c + K},
$$
$$
\rho = \frac{Z_2}{Z_1+Z_3},
$$
$$
c = \frac{r_s + N r_g}{N r_e}.
$$
{{< /math >}}

Here {{< math >}}$\rho${{< /math >}} is the ratio of the noise term to the other two and {{< math >}}$c${{< /math >}} is the fixed cost of a token, counted in experts per layer.

{{< livefig "moe-bound" `**Figure 8. At a fixed compute budget, the paper's bound puts the best K below M only through its noise term.** Each curve is divided by its own minimum, so 1.0 marks the best <i>K</i>. <i>M</i> = 16, as in the paper's language-model experiments. The sweep shows the best <i>K</i> sliding from <i>M</i> toward 1 as the noise ratio grows.` 430 `At a fixed compute budget, the paper's bound puts the best K below M only through its noise term` >}}

### What the bound does and does not establish

<span class="claim-tag claim-tag--d">derived</span> These points come from reading the appendix proofs line by line. The algebra behind the first, third, fourth and sixth is checked by code. The rest are readings of the paper.

- **The dependence on {{< math >}}$K${{< /math >}} comes from a loose step.** The proof bounds the gradient of each expert's own loss, averaged over the tokens routed to it, and adds that to the bound for the shared parameters. But the gradient of the full loss with respect to expert {{< math >}}$i${{< /math >}} is that quantity times the expert's token share {{< math >}}$\ell_i${{< /math >}}, which is about {{< math >}}$K/M${{< /math >}}. The sum is still a valid upper bound. On the expert part it is loose by {{< math >}}$1/\ell_i^2${{< /math >}} in squared norm, about {{< math >}}$(M/K)^2${{< /math >}} when the load is balanced, and the {{< math >}}$M/K${{< /math >}} in the factor {{< math >}}$1 + M/K${{< /math >}} comes from this step. With the exact conversion it becomes {{< math >}}$(M/K)(K/M)^2 = K/M${{< /math >}}, which points the other way.
- **The step-size cap depends on {{< math >}}$K${{< /math >}} and is then held constant.** The theorem needs the learning rate below a cap. Two of the cap's four terms carry a factor {{< math >}}$1/(NK)${{< /math >}}. In one of them the expert's token count, about {{< math >}}$BK/M${{< /math >}}, cancels the {{< math >}}$K${{< /math >}}, so one term falls as {{< math >}}$1/(NK)${{< /math >}}. The corollaries vary {{< math >}}$K${{< /math >}} with the base learning rate fixed.
- **The headline rate alone has no interior optimum.** With {{< math >}}$\rho = 0${{< /math >}} and {{< math >}}$c = 0${{< /math >}} the curve is {{< math >}}$\sqrt{K} + M/\sqrt{K}${{< /math >}}, which keeps falling until {{< math >}}$K = M${{< /math >}}. That is dense routing.
- **The optimum comes from the noise term.** With {{< math >}}$c = 0${{< /math >}} the minimum is at {{< math >}}$K^\ast = M/(1 + \rho)${{< /math >}}. The paper observes {{< math >}}$K^\ast = 2${{< /math >}} at {{< math >}}$M = 16${{< /math >}}, which corresponds to {{< math >}}$\rho = 7${{< /math >}}. Nothing measures {{< math >}}$\rho${{< /math >}}, so the bound fits any observed {{< math >}}$K^\ast${{< /math >}}.
- **The routing term is handled by assumption.** A corollary assumes each unselected expert's dense-softmax weight is at most {{< math >}}$\delta^t/M${{< /math >}}, and the theorem needs the {{< math >}}$\delta^t${{< /math >}} to sum to a constant over training. The appendix obtains that by supposing each step's bound on the routing discrepancy is no larger than that step's expected drop in loss. The corollary also states the inequality as ≤ where its proof writes ≥. Only ≤, applied to the unselected experts, gives the stated bound.
- **At {{< math >}}$K = 1${{< /math >}} the paper's own gate gives the router no gradient.** Its gate is a softmax over the selected experts, so Section 5 applies and the router's weights stay at their initialization. The paper describes no auxiliary loss and no routing noise, so as described its {{< math >}}$K = 1${{< /math >}} curves are runs whose router weights never leave their initialization.
- **The cost model fixes the size of an expert.** More experts per token then means more compute. Real designs hold the active parameters fixed and change the granularity, as in Section 3.
- **One measurement runs against the routing term.** In the paper's first figure, the loss gap to dense routing is smallest at {{< math >}}$K = 1${{< /math >}}, where the bound on it is largest. The paper notes this and answers that an upper bound is not contradicted. The experiments run from 3M to about 3B parameters, always with 8 or 16 experts and loss curves of 10,000 steps, and the best {{< math >}}$K${{< /math >}} at fixed compute is 2 in the language runs.

### What a toy says

<span class="claim-tag claim-tag--t">toy</span> Table 2 repeats the sweep in the tiny MoE, with compute proportional to {{< math >}}$K${{< /math >}}. The router input is centered, which keeps every expert alive at {{< math >}}$K \ge 2${{< /math >}}.

**Table 2. The tiny MoE at fixed compute: 8 experts stored, 6,000 ÷ {{< math >}}$K${{< /math >}} steps, mean loss over 8 seeds.**

| {{< math >}}$K${{< /math >}} | Steps | Loss, softmax over selected | Loss, Switch gate | Loss after 750 steps, softmax gate | Router |
|---|---:|---:|---:|---:|---|
| 1 | 6,000 | 0.68 | 0.73 | 1.43 | weights moved by exactly 0 with the softmax gate |
| 2 | 3,000 | 0.27 | 0.23 | 1.37 | trains |
| 4 | 1,500 | 0.66 | 0.63 | 1.44 | trains |
| 8 | 750 | 1.52 | 1.52 | 1.52 | trains, dense |

At equal compute {{< math >}}$K = 2${{< /math >}} wins, as in the paper. At equal steps more experts per token did not help: {{< math >}}$K = 8${{< /math >}} is the worst. A {{< math >}}$K = 1${{< /math >}} router that can learn, with the Switch gate, does no better on average than the frozen one, 0.73 ± 0.48 against 0.68 ± 0.16 across seeds, and it leaves 1.4 experts dead against 0.1. So in this toy, unfreezing the router did not remove the {{< math >}}$K = 1${{< /math >}} penalty.

## Appendix B: what was checked, and how

**Table 3. The derived claims, the check, and the result.**

| Claim | How it was checked | Result |
|---|---|---|
| MoE layer = wide FFN with a block mask | Both sides computed on 200 random layers | Largest difference {{< math >}}$7 \times 10^{-15}${{< /math >}} |
| Router gradient, and zero outside the top-{{< math >}}$K${{< /math >}} | Automatic differentiation, 300 random layers, log-cosh loss | Error below {{< math >}}$10^{-15}${{< /math >}}; exactly 0 outside |
| Policy-gradient form with mixture baseline | Same runs | Error below {{< math >}}$10^{-15}${{< /math >}} |
| Zero-sum inside the selected set | 300 random cases | Sum below {{< math >}}$2 \times 10^{-16}${{< /math >}} |
| No router gradient at {{< math >}}$K = 1${{< /math >}} with a renormalized gate | Automatic differentiation; 6,000 training steps of the toy | Exactly 0; router weights unchanged |
| Switch-gate update | Automatic differentiation, 300 cases | Error below {{< math >}}$10^{-15}${{< /math >}} |
| Time for an outsider to be tried | Direct simulation of the update at five gaps | 1,110 steps simulated, 1,112 predicted at {{< math >}}$\Delta = 3${{< /math >}} |
| Best {{< math >}}$K = M/(1 + \rho)${{< /math >}}; none inside when {{< math >}}$\rho = 0${{< /math >}} | Symbolic differentiation and grid search | Exact |
| Full gradient of an expert = token share × its own gradient | Automatic differentiation, 8 seeds of 8 experts | Difference below {{< math >}}$10^{-15}${{< /math >}} |
| One-layer sparse–dense gap ≤ 2 × norm × outside mass | 20,000 random cases | Never violated; tightest 0.87 |
| Expert signal-to-noise ∝ batch × token share | 300 minibatches, 120 experts in layers of 2 to 16 | Log–log slope 1.01 |

The toy is one MoE layer in NumPy with automatic differentiation: 96 token types in 8 clusters, 16-dimensional inputs, 8-dimensional targets, 8 experts of width 8, batch 128, Adam at {{< math >}}$3 \times 10^{-3}${{< /math >}}. Its results show that a mechanism can occur under real gradients. They say nothing about its size in a large model.

## Reading list

**The case against the design story**

- [Hash Layers For Large Sparse Models](https://arxiv.org/abs/2106.04426), Roller, Sukhbaatar, Szlam, Weston, NeurIPS 2021. Tables 1 and 3 are the ones to read.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088), Jiang et al., 2024. Section 5, routing analysis.
- [OpenMoE](https://arxiv.org/abs/2402.01739), Xue et al., ICML 2024. Section 4: token-level routing, early lock-in, dropped tokens.
- [ST-MoE](https://arxiv.org/abs/2202.08906), Zoph et al., 2022. Section 7 on encoder and decoder experts.

**What experts do**

- [The Expert Strikes Back](https://arxiv.org/abs/2604.02178), Herbst, Wermter, Lee, ICML 2026.
- [RouterInterp: Understanding Superposed Specialisation in Mixture of Experts Routing](https://arxiv.org/abs/2610.11775), Lasy, Cai, Ayonrinde, ICML 2026.
- [The Illusion of Specialization: Unveiling the Domain-Invariant “Standing Committee” in Mixture-of-Experts Models](https://arxiv.org/abs/2601.03425), Wang et al., ACL 2026.
- [Probing Semantic Routing in Large Mixture-of-Expert Models](https://arxiv.org/abs/2502.10928), Olson et al., 2025.
- [Sparsity and Superposition in Mixture of Experts](https://arxiv.org/abs/2510.23671), Chaudhari, Nuer, Thorstenson, NeurIPS 2025 workshop.

**Training dynamics**

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), Shazeer et al., 2017. Section 4 on self-reinforcing imbalance.
- [Switch Transformers](https://arxiv.org/abs/2101.03961), Fedus, Zoph, Shazeer, 2021. Section 2 and Appendix C on exploration.
- [Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](https://arxiv.org/abs/2504.12463), Panda et al., NeurIPS 2025.
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664), Wang et al., 2024, and the [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437).
- [Rethinking Convergence in MoE Training: The Role of Routing Sparsity](https://openreview.net/forum?id=dNLXIVeCqn), Zhu et al., ICML 2026.
- [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370), Ma et al., ICML 2026.

**Scaling, design, theory**

- [Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/abs/2402.07871), Krajewski et al., 2024.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066), Dai et al., 2024.
- [Towards Understanding the Mixture-of-Experts Layer in Deep Learning](https://arxiv.org/abs/2208.02813), Chen, Deng, Wu, Gu, Li, NeurIPS 2022.
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913), Geva et al., EMNLP 2021.

**Introductions**

- Cameron Wolfe: [Mixture-of-Experts (MoE): The Birth and Rise of Conditional Computation](https://cameronrwolfe.substack.com/p/conditional-computation-the-birth), [Mixture-of-Experts (MoE) LLMs](https://cameronrwolfe.substack.com/p/moe-llms), and [nanoMoE](https://cameronrwolfe.substack.com/p/nano-moe), a from-scratch PyTorch implementation.
- Maarten Grootendorst: [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts).

*Sources checked 27 November 2026.*
