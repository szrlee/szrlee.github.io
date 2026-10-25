---
title: "Sparse in the Vocabulary, Low-Rank in Hidden Space"
subtitle: "The Shape of a Post-Training Update, Part 4: the tokens that compete at forks bound the rank of the update"
summary: "At the output layer only the rows of tokens that compete can move. The signal that backpropagation carries into the network is the drawn token's output vector minus the sampler's average, so it lies in the span of differences of those vectors and vanishes where the sampler is sure. A linear map that feeds the unembedding therefore gets an expected update of rank at most the number of competing tokens minus one, plus a leak set by the tail mass, and a LoRA of that rank carries the main part. Measured updates are numerically full rank yet spectrally concentrated; the optimizer and many steps account for the difference."
authors:
  - admin
tags:
  - Reinforcement Learning
  - Policy Gradient
  - Fisher Information
  - Sparsity
  - Low Rank
  - LoRA
  - Language Models
categories:
  - Research
  - Theory
date: '2026-10-24T00:00:00Z'
lastmod: '2026-10-24T00:00:00Z'
math: true
draft: false
series:
  - The Shape of a Post-Training Update
series_order: 4
---

{{% callout note %}}
**The Shape of a Post-Training Update · Part 4 of 4.** Follows the hidden-layer Fisher of [Part 2](/post/sparsity-score-centering/#2-hidden-layers-a-few-directions) from a cap on each parameter to the rank of a whole update.

- **The same tokens, in two bases.** At the output layer only the rows of tokens that compete can move: sparsity in the vocabulary. Inside the network those tokens appear through their output vectors, and the update is confined to the few directions that tell them apart: low rank in hidden space.
- **Only forks send signal.** Backpropagation carries the drawn token's output vector minus the sampler's average. Its expected size shrinks to zero as the sampler's top token approaches probability one, so positions where the sampler is sure contribute almost nothing.
- **The head's update is low-rank, up to a leak.** For a linear map that feeds the unembedding, the expected update is a part of rank at most the number of tokens competing at forks minus one, plus a leak set by the sampler's mass outside those tokens. A LoRA (low-rank adapter) of that rank carries the main part.
- **Below the head the same subspace travels along Jacobian paths.** The final normalization adds one direction per position; how many paths carry signal is an empirical quantity.
- **The size of each coefficient gives LoRA's shape.** With a Kronecker-factored Fisher, the cap of [Part 1](/post/gradient-is-covariance/) applies to every pair of eigen-directions, and the update sits in a small block: a tall factor times a flat one.
- **Measured updates are full rank and still concentrated.** Accumulated updates after reinforcement learning (RL) have full numerical rank, while a few singular directions hold much of their energy, and rank-one LoRA matches full RL. The optimizer, the drift of the forks and noise separate one expected step from the accumulated change.
{{% /callout %}}

**Starting point.** A prefix {{< math >}}$s_t=(x,y_{\lt t})${{< /math >}} has a last hidden state {{< math >}}$h(s_t)\in\mathbb{R}^d${{< /math >}}, and each token {{< math >}}$a${{< /math >}} of the vocabulary {{< math >}}$\mathcal V${{< /math >}} has logit {{< math >}}$z_a=w_a^\top h${{< /math >}}, where the output vector {{< math >}}$w_a${{< /math >}} is its row of the unembedding; write {{< math >}}$B${{< /math >}} for a bound on the row norms, {{< math >}}$\|w_a\|\le B${{< /math >}}. The sampler {{< math >}}$\smu${{< /math >}} drew the response, and {{< math >}}$A${{< /math >}} is the learning signal, a reward or advantage. Write {{< math >}}$\kappa_{i,a}=\partial z_a/\partial\theta_i${{< /math >}} for how strongly parameter {{< math >}}$i${{< /math >}} moves the logit of token {{< math >}}$a${{< /math >}}; the score centered under the sampler is {{< math >}}$c_i=\sum_t c_i^{(t)}${{< /math >}} with {{< math >}}$c_i^{(t)}=\kappa_{i,y_t}-\sum_a\smu(a\mid s_t)\,\kappa_{i,a}${{< /math >}}. With it, the expected update of parameter {{< math >}}$i${{< /math >}} is a covariance capped by the Fisher ([Part 1](/post/gradient-is-covariance/)):

{{< math >}}
$$
G_i=\operatorname{Cov}_{\smu}(A,c_i),\qquad |G_i|\le\operatorname{std}_{\smu}(A)\sqrt{F^{\mu}_i}.
$$
{{< /math >}}

In hidden layers, with {{< math >}}$u_i=\partial h/\partial\theta_i${{< /math >}}, the token Fisher at prefix {{< math >}}$s${{< /math >}} is a quadratic form in the covariance of the output vectors under the sampler, and that covariance has rank at most {{< math >}}$n-1${{< /math >}} when the sampler gives {{< math >}}$n${{< /math >}} tokens positive probability ([Part 2, section 2](/post/sparsity-score-centering/#2-hidden-layers-a-few-directions)):

{{< math >}}
$$
f_i(s)=u_i^\top C_\mu(s)\,u_i,\qquad C_\mu=\sum_a\smu(a)\,(w_a-\bar w_\mu)(w_a-\bar w_\mu)^\top,\qquad \bar w_\mu=\sum_a\smu(a)\,w_a .
$$
{{< /math >}}

Part 2 used this to cap each coordinate. Here it bounds whole matrices: the same few directions that keep the Fisher small also bound the rank of the update.

## 1. The backward signal lives at the forks

**Backpropagation carries the drawn token's output vector minus the sampler's average, and that vector is nearly zero wherever the sampler is sure.**

{{< stepfig "lr-forks" >}}

At the output layer the parameters are indexed by the vocabulary: an entry of the unembedding row of token {{< math >}}$c${{< /math >}} moves the logit of {{< math >}}$c${{< /math >}} alone, its token Fisher is {{< math >}}$h_k^2\,\smu(c)(1-\smu(c))${{< /math >}}, and the rows of tokens that never compete cannot move ([Part 2, section 1](/post/sparsity-score-centering/#1-output-layer-an-exact-coin-flip)). Every parameter below the unembedding reaches the logits through {{< math >}}$h${{< /math >}}, so {{< math >}}$\kappa_{i,a}=w_a^\top u_i${{< /math >}}, and its centered score at position {{< math >}}$t${{< /math >}} is one vector pulled back through {{< math >}}$u_i${{< /math >}}:

{{< math >}}
$$
\delta_t:=w_{y_t}-\bar w_\mu(s_t),\qquad c_i^{(t)}=\delta_t^\top u_i(s_t).
$$
{{< /math >}}

The vector {{< math >}}$\delta_t${{< /math >}} is the same for every parameter. Given the prefix, it has mean zero and covariance {{< math >}}$C_\mu(s_t)${{< /math >}}, and it is one of the centered output vectors, so it lies in the range of {{< math >}}$C_\mu(s_t)${{< /math >}}. Its expected squared size is a variance, at most the second moment about the most likely token's vector {{< math >}}$w_{a^\star}${{< /math >}}; that token contributes zero and every other difference has norm at most {{< math >}}$2B${{< /math >}}:

{{< math >}}
$$
\mathbb{E}\big[\|\delta_t\|^2\mid s_t\big]=\operatorname{tr}C_\mu(s_t)\le\sum_a\smu(a\mid s_t)\,\|w_a-w_{a^\star}\|^2\le4B^2\Big(1-\max_a\smu(a\mid s_t)\Big).
$$
{{< /math >}}

Where the sampler is sure, the backward signal is nearly zero. Along a rollout it is flat except at forks, the few positions where several tokens compete, and at a fork with {{< math >}}$n${{< /math >}} competitors it spans at most {{< math >}}$n-1${{< /math >}} directions. A single draw can still be large: for a rare token {{< math >}}$\|\delta_t\|${{< /math >}} can reach {{< math >}}$2B${{< /math >}}, and the bound holds on average over the draw.

## 2. The head's update is low-rank, up to a leak

**For a linear map that produces the unembedding input, the expected update splits into a part whose rank is set by the tokens that compete at forks, and a leak set by the sampler's mass outside them.**

{{< stepfig "lr-leak" >}}

Let {{< math >}}$M${{< /math >}} be a linear map with {{< math >}}$h(s_t)=M x_t${{< /math >}}, where the input {{< math >}}$x_t${{< /math >}} is computed before {{< math >}}$M${{< /math >}} and so does not depend on it. (In a transformer a final normalization sits between the last linear layer and the unembedding; [section 4](#4-below-the-head) adds it.) For every prefix {{< math >}}$s${{< /math >}} choose a candidate set {{< math >}}$K_s${{< /math >}} of tokens, and write

{{< math >}}
$$
\rho_s=\smu\big(\mathcal V\setminus K_s\mid s\big),
$$
$$
D=\operatorname{span}\big\{w_a-w_b:\ a,b\in K_s\ \text{for some prefix } s\big\}
$$
{{< /math >}}

for the sampler's mass outside the candidates and the span of differences of candidate output vectors, and {{< math >}}$P${{< /math >}} for the orthogonal projector onto {{< math >}}$D${{< /math >}}. Then the expected update {{< math >}}$G_M=\mathbb{E}_{\smu}\big[A\sum_t\delta_t x_t^\top\big]${{< /math >}} splits as {{< math >}}$G_M=PG_M+(I-P)G_M${{< /math >}}, with

{{< math >}}
$$
\operatorname{rank}(PG_M)\le\dim D\le|K_{\mathrm{fork}}|-1,\qquad K_{\mathrm{fork}}=\bigcup_{|K_s|\ge2}K_s,
$$
$$
\big\|(I-P)\,G_M\big\|_F\le2\sqrt2\;\operatorname{std}_{\smu}(A)\;B\,\Big(\mathbb{E}_{\smu}\sum_t\rho_{s_t}\,\|x_t\|^2\Big)^{1/2}.
$$
{{< /math >}}

The candidate sets are a free choice. At a fork, take the competing tokens. At a confident prefix, take the top token alone: one vector has no differences, so it adds no direction, and its leak {{< math >}}$\rho_s=1-\max_a\smu(a\mid s)${{< /math >}} is small exactly there. Larger candidate sets lower the leak and raise the rank. The statement concerns the expected update at a fixed sampler; it makes no assumption about the inputs {{< math >}}$x_t${{< /math >}} and needs no independence between the backward signal and the inputs.

<details class="spoiler">
<summary>Proof</summary>

**The update.** The centered score at position {{< math >}}$t${{< /math >}} depends on {{< math >}}$M${{< /math >}} only through {{< math >}}$h(s_t)=Mx_t${{< /math >}}, so its gradient with respect to {{< math >}}$M${{< /math >}} is {{< math >}}$\delta_t x_t^\top${{< /math >}}. Since {{< math >}}$\mathbb{E}[\delta_t\mid s_t]=0${{< /math >}} and {{< math >}}$x_t${{< /math >}} is a function of {{< math >}}$s_t${{< /math >}}, {{< math >}}$\mathbb{E}[\delta_t x_t^\top]=0${{< /math >}}, and {{< math >}}$A${{< /math >}} can be replaced by {{< math >}}$A-\mathbb{E}A${{< /math >}}:

{{< math >}}
$$
G_M=\mathbb{E}_{\smu}\Big[(A-\mathbb{E}A)\sum_t\delta_t x_t^\top\Big].
$$
{{< /math >}}

**The rank.** Every column of {{< math >}}$PG_M${{< /math >}} lies in {{< math >}}$D${{< /math >}}, so {{< math >}}$\operatorname{rank}(PG_M)\le\dim D${{< /math >}}. For a fixed {{< math >}}$a_0\in K_{\mathrm{fork}}${{< /math >}}, {{< math >}}$w_a-w_b=(w_a-w_{a_0})-(w_b-w_{a_0})${{< /math >}}, so {{< math >}}$D${{< /math >}} lies in the span of the {{< math >}}$|K_{\mathrm{fork}}|-1${{< /math >}} vectors {{< math >}}$w_a-w_{a_0}${{< /math >}}; candidate sets with one token contribute no differences.

**The leak.** The norm of an expectation is at most the expectation of the norm, and Cauchy–Schwarz splits off the signal:

{{< math >}}
$$
\big\|(I-P)G_M\big\|_F\le\operatorname{std}_{\smu}(A)\,\Big(\mathbb{E}\Big\|\sum_t(I-P)\,\delta_t x_t^\top\Big\|_F^2\Big)^{1/2}.
$$
{{< /math >}}

For outer products, {{< math >}}$\langle ab^\top,cd^\top\rangle_F=(a^\top c)(b^\top d)${{< /math >}}. For {{< math >}}$u\lt t${{< /math >}} the cross term contains {{< math >}}$\mathbb{E}[\delta_t\mid s_t]=0${{< /math >}}, because {{< math >}}$\delta_u${{< /math >}}, {{< math >}}$x_u${{< /math >}} and {{< math >}}$x_t${{< /math >}} are fixed once {{< math >}}$s_t${{< /math >}} is known, so only the diagonal survives, as in [Part 1](/post/gradient-is-covariance/#4-along-the-response-the-fisher-adds-up):

{{< math >}}
$$
\mathbb{E}\Big\|\sum_t(I-P)\,\delta_t x_t^\top\Big\|_F^2=\sum_t\mathbb{E}\Big[\|x_t\|^2\;\mathbb{E}\big[\|(I-P)\,\delta_t\|^2\mid s_t\big]\Big].
$$
{{< /math >}}

**One position.** Write {{< math >}}$\delta_t=\sum_a\smu(a)\,(w_{y_t}-w_a)${{< /math >}}. If the drawn token is a candidate, {{< math >}}$I-P${{< /math >}} removes every term with {{< math >}}$a\in K_s${{< /math >}}, and the rest has total weight {{< math >}}$\rho_s${{< /math >}} and norm at most {{< math >}}$2B\rho_s${{< /math >}}. If it is not, which happens with probability {{< math >}}$\rho_s${{< /math >}}, {{< math >}}$\|\delta_t\|\le2B${{< /math >}}. Adding the two cases,

{{< math >}}
$$
\mathbb{E}\big[\|(I-P)\,\delta_t\|^2\mid s_t\big]\le4B^2\rho_s^2+4B^2\rho_s\le8B^2\rho_s,
$$
{{< /math >}}

and the three displays combine into the bound.

</details>

## 3. What this says about LoRA

**An adapter of rank {{< math >}}$\dim D${{< /math >}} carries the head's expected update up to the leak, and gradient descent on LoRA finds that subspace by itself.**

Because {{< math >}}$PG_M${{< /math >}} has rank at most {{< math >}}$\dim D${{< /math >}}, the best approximation of {{< math >}}$G_M${{< /math >}} by a matrix of rank {{< math >}}$r\ge\dim D${{< /math >}} is at least as good as {{< math >}}$PG_M${{< /math >}} ([Eckart & Young, 1936](https://doi.org/10.1007/BF02288367), in its easy direction):

{{< math >}}
$$
\min_{\operatorname{rank}X\le r}\big\|G_M-X\big\|_F\le\big\|(I-P)\,G_M\big\|_F,\qquad PG_M=U\,(G_M^\top U)^\top=U\,V^\top,
$$
{{< /math >}}

with {{< math >}}$U${{< /math >}} an orthonormal basis of {{< math >}}$D${{< /math >}} and {{< math >}}$V=G_M^\top U${{< /math >}}. That is the shape of LoRA, an update {{< math >}}$\Delta M=UV^\top${{< /math >}} ([Hu et al., 2021](https://arxiv.org/abs/2106.09685), who write it {{< math >}}$BA${{< /math >}}). Here both factors have a meaning: {{< math >}}$U${{< /math >}} is built from differences of the competing tokens' output vectors, and {{< math >}}$V${{< /math >}} records what the inputs look like at the forks.

LoRA reaches this subspace on its own. With the factor {{< math >}}$U${{< /math >}} initialized at zero and {{< math >}}$V${{< /math >}} at random, the first gradient on {{< math >}}$U${{< /math >}} is {{< math >}}$G_MV_0${{< /math >}}, whose columns lie in the column space of {{< math >}}$G_M${{< /math >}}, inside {{< math >}}$D${{< /math >}} up to the leak, while {{< math >}}$V${{< /math >}} gets no gradient in that step.

How large {{< math >}}$\dim D${{< /math >}} is depends on the task. When the forks are the same few decisions in rollout after rollout, with the same candidates, {{< math >}}$D${{< /math >}} is small. For a broad task the union of competing tokens over all prompts can run into the thousands, and the bound grows with it. LoRA of rank one matches full fine-tuning in RL on math ([Schulman & Thinking Machines Lab, 2025](https://thinkingmachines.ai/blog/lora/)), so the useful part of the update can be much smaller than this count. Two further arguments explain that: the signal itself carries little, at most one bit per pass/fail episode ([Information Bandwidth](/post/information-bandwidth-rl/)), and the Fisher weighs the directions unevenly ([section 5](#5-the-size-of-each-coefficient-gives-loras-shape)).

## 4. Below the head

**Below the head the same subspace is carried down along Jacobian paths; the final normalization adds one direction per position.**

{{< stepfig "lr-paths" >}}

For a matrix {{< math >}}$M_\ell${{< /math >}} in a lower layer, write {{< math >}}$o_\tau=M_\ell x_\tau${{< /math >}} for its output at position {{< math >}}$\tau${{< /math >}} and {{< math >}}$J_{t,\tau}=\partial h(s_t)/\partial o_\tau${{< /math >}} for the Jacobian that carries it to the last hidden state at a later position {{< math >}}$t${{< /math >}}. The gradient of the centered score sums over the earlier positions:

{{< math >}}
$$
\frac{\partial c^{(t)}}{\partial M_\ell}=\sum_{\tau\le t}\big(J_{t,\tau}^\top\delta_t\big)\,x_\tau^\top,\qquad \operatorname{col}(\text{main part})\subseteq\sum_{(t,\tau)}J_{t,\tau}^\top D .
$$
{{< /math >}}

Each image {{< math >}}$J_{t,\tau}^\top D${{< /math >}} has dimension at most {{< math >}}$\dim D${{< /math >}}, so the rank of the main part is at most {{< math >}}$\dim D${{< /math >}} times the number of distinct maps {{< math >}}$J_{t,\tau}^\top${{< /math >}} that carry signal. The bound is exact. The number of such maps is an empirical quantity; attention that concentrates on a few positions keeps it small.

In a transformer the last hidden state is a normalized residual stream, {{< math >}}$h=g\odot r/\operatorname{rms}(r)${{< /math >}} with a learned gain {{< math >}}$g${{< /math >}} ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)). With {{< math >}}$\hat r=r/\|r\|${{< /math >}} and {{< math >}}$\delta_t${{< /math >}} in {{< math >}}$D${{< /math >}}, as it is for the main part, the signal that reaches the residual stream at position {{< math >}}$t${{< /math >}} is

{{< math >}}
$$
\Big(\frac{\partial h}{\partial r}\Big)^{\!\top}\delta_t=\frac{1}{\operatorname{rms}(r_t)}\Big(\operatorname{diag}(g)\,\delta_t-\big(\hat r_t^\top\operatorname{diag}(g)\,\delta_t\big)\,\hat r_t\Big)\ \in\ \operatorname{diag}(g)\,D+\operatorname{span}\{\hat r_t\}:
$$
{{< /math >}}

the rescaled subspace, plus one direction per position, the position's own residual direction, weighted by its overlap with the signal. Every layer adds its output to the residual stream, so the identity path hands this signal to the output of every layer; the Jacobians of attention and of the feed-forward blocks add the other paths.

## 5. The size of each coefficient gives LoRA's shape

**Counting directions gives a rank; weighing them by the Fisher shows where the update's energy sits, and with a Kronecker-factored Fisher that place is a small block.**

{{< stepfig "lr-kfac" >}}

For any unit direction {{< math >}}$w${{< /math >}} in parameter space, the Cauchy–Schwarz step of [Part 1](/post/gradient-is-covariance/) gives {{< math >}}$|w^\top G|\le\operatorname{std}_{\smu}(A)\sqrt{w^\top F^{\mu}w}${{< /math >}}; the per-parameter cap is the case where {{< math >}}$w${{< /math >}} is one coordinate. Along the eigenvectors {{< math >}}$u_j${{< /math >}} of {{< math >}}$F^{\mu}${{< /math >}}, with eigenvalues {{< math >}}$\lambda_j${{< /math >}}, the coefficients are capped by {{< math >}}$\sqrt{\lambda_j}${{< /math >}}, and everything outside the top {{< math >}}$k${{< /math >}} directions is bounded by the tail of the spectrum:

{{< math >}}
$$
\big\|G-P_kG\big\|^2=\sum_{j>k}\big(u_j^\top G\big)^2\le\operatorname{Var}_{\smu}(A)\sum_{j>k}\lambda_j .
$$
{{< /math >}}

Take the head's map {{< math >}}$M${{< /math >}} of section 2, with input {{< math >}}$x_t${{< /math >}} and backward signal {{< math >}}$\delta_t${{< /math >}}; its gradient at position {{< math >}}$t${{< /math >}} is {{< math >}}$\delta_t x_t^\top${{< /math >}}. Cross terms between positions vanish as in the proof of section 2, so its Fisher is exactly {{< math >}}$F_M=\mathbb{E}_{\smu}\sum_t(x_tx_t^\top)\otimes(\delta_t\delta_t^\top)${{< /math >}}. The Kronecker-factored approximation, K-FAC ([Martens & Grosse, 2015](https://arxiv.org/abs/1503.05671)), treats {{< math >}}$\delta${{< /math >}} and {{< math >}}$x${{< /math >}} as independent and replaces each term by a product of averages over prefixes and positions. The Fisher then acts on a matrix {{< math >}}$X${{< /math >}} from both sides, and its eigen-directions are outer products of the eigenvectors {{< math >}}$p_i${{< /math >}} of {{< math >}}$S${{< /math >}} and {{< math >}}$q_j${{< /math >}} of {{< math >}}$H${{< /math >}}:

{{< math >}}
$$
F_M\operatorname{vec}(X)\approx T\operatorname{vec}(S\,X\,H),\qquad S=\overline{\delta\,\delta^\top},\quad H=\overline{x\,x^\top},
$$
$$
\big|p_i^\top G_M\,q_j\big|\le\operatorname{std}_{\smu}(A)\sqrt{T\,\sigma_i\,\eta_j},
$$
{{< /math >}}

where the bars average over prefixes and positions, {{< math >}}$T${{< /math >}} is the number of positions, and {{< math >}}$\sigma_i${{< /math >}} and {{< math >}}$\eta_j${{< /math >}} are the eigenvalues of {{< math >}}$S${{< /math >}} and {{< math >}}$H${{< /math >}}. Lower layers fit the same form with their own backward signal, and there K-FAC is an approximation from the start. If {{< math >}}$\sigma_i${{< /math >}} is small beyond the first {{< math >}}$r_s${{< /math >}} and {{< math >}}$\eta_j${{< /math >}} beyond the first {{< math >}}$r_h${{< /math >}}, every coefficient outside an {{< math >}}$r_s\times r_h${{< /math >}} block is small, and in these bases the update is {{< math >}}$G_M\approx P_r\,\Gamma\,Q_r^\top${{< /math >}}, of rank at most {{< math >}}$\min(r_s,r_h)${{< /math >}}: a tall factor {{< math >}}$P_r${{< /math >}} of the top eigenvectors of {{< math >}}$S${{< /math >}}, a small block {{< math >}}$\Gamma${{< /math >}}, and a flat factor {{< math >}}$Q_r^\top${{< /math >}} of the top eigenvectors of {{< math >}}$H${{< /math >}}. The tall factor spans the output directions the layer is most sensitive to; the flat one spans the input directions it sees most.

At the head, {{< math >}}$\mathbb{E}[\delta_t\delta_t^\top\mid s_t]=C_\mu(s_t)${{< /math >}}, so {{< math >}}$S${{< /math >}} is the average of the covariances of [Part 2](/post/sparsity-score-centering/#2-hidden-layers-a-few-directions) along the rollouts, large only at forks. The two arguments meet there. Counting gives the rank bound of section 2 without any factorization; weighing explains why a rank far below that bound can suffice. At the unembedding itself the Kronecker form holds exactly at each prefix: its token Fisher is {{< math >}}$\big(\operatorname{diag}(\smu)-\smu\smu^\top\big)\otimes hh^\top${{< /math >}} ([Part 2, section 1](/post/sparsity-score-centering/#1-output-layer-an-exact-coin-flip)).

## 6. What the measurements show

**Accumulated updates have full numerical rank and still keep a large share of their energy in a few directions; the bounds above describe one expected step, and three things separate it from the accumulated change.**

The measurements:

- **Few coordinates, full numerical rank.** RL changes 5–30% of the parameters, and the update to almost every matrix is nearly full rank ([Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)). Part of this sparsity is precision: a change too small to alter a bfloat16 weight does not register ([Zhu et al., 2025](https://arxiv.org/abs/2511.08567)).
- **Full rank, concentrated energy.** Across six runs of on-policy distillation, the median matrix has full numerical rank at a relative threshold of {{< math >}}$10^{-5}${{< /math >}}, while its top 16 singular values hold 20–41% of the energy and its stable rank is 7–20. Two RL runs with verifiable rewards look the same (top-16 energy 19.6% and 26.9%, stable rank 16.1 and 12.4); offline distillation, which trains on another model's outputs, is diffuse (8.6% and 82.7) ([Yu et al., 2026](https://arxiv.org/abs/2606.13657)).
- **The subspace forms early.** Over training the stable rank of the cumulative update contracts for RL with verifiable rewards, stays in a narrow band for on-policy distillation, and expands for supervised fine-tuning (SFT); restricting on-policy distillation to the subspace of its early updates preserves its performance ([Shen et al., 2026](https://arxiv.org/abs/2606.07082)).
- **Low capacity is enough.** Rank-one LoRA matches full fine-tuning in RL ([Schulman & Thinking Machines Lab, 2025](https://thinkingmachines.ai/blog/lora/)).
- **The optimizer shapes the support.** With stochastic gradient descent (SGD), full fine-tuning in RL changes fewer than 0.02% of the parameters, more than 1,000 times fewer than with AdamW, and matches or beats its accuracy ([Mukherjee et al., 2026](https://arxiv.org/abs/2602.07729)).

What separates one expected step from the accumulated change:

- **The optimizer.** An SGD step is proportional to the gradient, so the cap of Part 1 and the subspace of section 2 apply to its expected step. Adam divides each entry by its own running scale, which is not a linear map on the matrix: its first step is close to the entrywise sign of the gradient, and the sign pattern of a matrix of rank two or more is generally full rank. Small coordinates are lifted toward the step size of large ones, which spreads the change over more coordinates and more directions.
- **The forks move.** As training sharpens the sampler, the competing tokens and their output vectors change, and so does {{< math >}}$D${{< /math >}}. The initial high-Fisher coordinates keep 70–78% of their membership over 1,000 steps ([Zhang, Mukherjee & Peng, 2026](https://arxiv.org/abs/2609.36262)), so the set of directions that the steps use grows over training.
- **Noise and the leak fill every direction weakly.** Numerical rank counts singular values above {{< math >}}$10^{-5}${{< /math >}} of the largest, so a weak floor of noise makes any accumulated matrix full rank. Stable rank and energy fractions measure where the update's weight lies, which is what the bounds describe.

## 7. Supervised fine-tuning leaves the forks

**Without centering, the gradient of supervised fine-tuning points toward whichever token the data demonstrates, at every position where data and model disagree, so its directions add up position by position.**

At prefix {{< math >}}$s${{< /math >}}, the gradient of the log-likelihood with respect to the unembedding input is {{< math >}}$w_{y}-\bar w_p(s)${{< /math >}} for a demonstrated token {{< math >}}$y${{< /math >}}, where {{< math >}}$\bar w_p${{< /math >}} is the model's average output vector. Averaged over the data distribution {{< math >}}$q${{< /math >}} it is a shift of the mean:

{{< math >}}
$$
\mathbb{E}_{y\sim q}\big[w_y-\bar w_p(s)\big]=\bar w_q(s)-\bar w_p(s).
$$
{{< /math >}}

This is the mean shift of [Part 2, section 3](/post/sparsity-score-centering/#3-without-centering-the-mean-shift), read in hidden space. It is nonzero wherever the data disagrees with the model, including prefixes where the model is sure, and there it points from the model's top token toward the demonstrated one, which need not be among the model's candidates. Confident positions, which send almost nothing in on-policy policy gradient, contribute in full, and each contributes its own direction. The measurements agree: in a controlled comparison on Qwen3-8B, 8.1% of the weights are unchanged after SFT, against 51.6% after on-policy distillation and 77.2% after RL with verifiable rewards, counted at bfloat16 precision ([Shen et al., 2026](https://arxiv.org/abs/2606.07082)), and offline distillation has the most diffuse spectrum of the runs above ([Yu et al., 2026](https://arxiv.org/abs/2606.13657)). On-policy distillation samples from the model and centers its score the same way as RL, and it falls between the two.

## 8. Scope and open questions

- **Where it is exact.** For the unembedding input and for a linear map that produces it. Through the final normalization it picks up one direction per position; below that, the number of Jacobian paths that carry signal is measured.
- **One step, first order.** Every bound here is for the expected update at a fixed sampler. The forks drift during training, and a bound on the accumulated change would have to control that drift.
- **The optimizer.** For SGD the subspace argument applies to the expected step. For Adam it applies to the gradient, and the per-coordinate scaling raises both the rank and the support of the step.
- **Temperature.** At temperature one every token has positive probability, so formally the whole vocabulary competes at every prefix; the leak term absorbs the tail, and top-p sampling makes the candidate sets exact.
- **What to measure.** The spectrum of {{< math >}}$S${{< /math >}}, the average of {{< math >}}$C_\mu(s_t)${{< /math >}} along real rollouts; the number of distinct forks per task; and the overlap between the top singular vectors of a measured head update and the span of differences of the competing tokens' output vectors.

**The series in one line.** A score centered under the sampler makes the update a covariance capped by the Fisher ([Part 1](/post/gradient-is-covariance/)); that Fisher is concentrated when the sampler favors a few tokens that react alike, and without centering a mean shift escapes it ([Part 2](/post/sparsity-score-centering/)); a response moves only as far as it resembles what the sampler produces ([Part 3](/post/what-policy-gradient-can-reach/)); and the same competing tokens make the update sparse in the vocabulary and low-rank in hidden space (this part).

## References

- Y. Zhang, S. Mukherjee, H. Peng. [Understanding LLM Parameter Update Sparsity through the Lens of Fisher](https://arxiv.org/abs/2609.36262). arXiv, 2026.
- S. Mukherjee, L. Yuan, D. Hakkani-Tür, H. Peng. [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711). NeurIPS 2025.
- S. Mukherjee, L. Yuan, P. Jayasinha, D. Hakkani-Tür, H. Peng. [Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs](https://arxiv.org/abs/2602.07729). arXiv, 2026.
- G. Yu, W. Liu, Y. Hu, et al. [Dense Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation](https://arxiv.org/abs/2606.13657). arXiv, 2026.
- Z. Shen, Y. Li, Q. Yin, et al. [On the Geometry of On-Policy Distillation](https://arxiv.org/abs/2606.07082). arXiv, 2026.
- H. Zhu, Z. Zhang, H. Huang, et al. [The Path Not Taken: RLVR Provably Learns Off the Principals](https://arxiv.org/abs/2511.08567). arXiv, 2025.
- J. Schulman and Thinking Machines Lab. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/). Thinking Machines Lab: Connectionism, 2025.
- E. J. Hu, Y. Shen, P. Wallis, et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR 2022.
- J. Martens, R. Grosse. [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671). ICML 2015.
- C. Eckart, G. Young. [The Approximation of One Matrix by Another of Lower Rank](https://doi.org/10.1007/BF02288367). Psychometrika, 1936.
- B. Zhang, R. Sennrich. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). NeurIPS 2019.
- Y. Li. [Information Bandwidth in Reinforcement Learning](/post/information-bandwidth-rl/). Blog, 2025.

## Citation

```bibtex
@online{li-2026-low-rank-hidden-space,
  title  = {Sparse in the Vocabulary, Low-Rank in Hidden Space},
  author = {Li, Yingru},
  year   = {2026},
  month  = oct,
  url    = {https://richardli.xyz/post/low-rank-hidden-space/}
}
```
