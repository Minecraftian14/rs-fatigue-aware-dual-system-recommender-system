> [!question]- How exactly are we going to utilize the fatigue metric? Like, how is it incorporated in the algorithm and why do we need it?

Excellent — this is the *right* question.

If fatigue is just a “nice idea,” reviewers will destroy it.
You need:

1. A **mathematical definition**
2. A **clear algorithmic role**
3. A **justification of necessity**
4. Evidence it changes behavior

Let’s make it concrete.

---

# 1️⃣ Why Do We Even Need a Fatigue Metric?

Standard recommender systems assume:

> More engagement = more satisfaction.

But in reality:

* Repeated similar items → boredom
* Long sessions → cognitive exhaustion
* Overexposure → reduced novelty value
* Continuous dopamine spikes → short-term retention but long-term churn

Without modeling fatigue, your model will:

* Over-recommend similar high-click items
* Encourage binge behavior
* Inflate short-term metrics
* Hurt long-term retention

Fatigue is the **missing state variable** in sequential recommendation.

It captures diminishing marginal utility.

---

# 2️⃣ What Is Fatigue (Formally)?

We define fatigue as a latent dynamic state variable:

[
F_t = f(H_t)
]

Where:

* (H_t) = recent interaction history
* (F_t) ∈ [0,1]
* Higher value = higher cognitive burden / saturation

It evolves over time:

[
F_t = \lambda F_{t-1} + g(similarity, session_length, repetition, time_gap)
]

Where:

* λ = decay factor
* g(.) increases fatigue when:

  * Items are similar
  * Session is long
  * Repetition is high

Fatigue naturally decays when:

* User takes a break
* Content diversity increases

---

# 3️⃣ How Is It Incorporated Into the Algorithm?

Now we get precise.

There are three integration strategies.

---

## 🔹 Method 1: Feature-Level Integration (Simplest)

Add fatigue as an input feature to your ranking model.

Instead of:

[
Score(u,i)
]

You compute:

[
Score(u,i, F_t)
]

This lets the model learn:

* When fatigue is high → promote novelty
* When fatigue is low → allow engagement optimization

This is easiest and publishable.

---

## 🔹 Method 2: Reward-Level Integration (Stronger)

In RL-based recommendation:

Original reward:

[
R_t = Click_t
]

Your modified reward:

[
R_t = \alpha Click_t - \beta F_t
]

Now the agent learns:

* Maximize engagement
* Minimize fatigue accumulation

This explicitly penalizes overconsumption.

This operationalizes long-term optimization.

---

## 🔹 Method 3: Gating Mechanism (System 1 vs System 2)

Define a switching function:

If:
[
F_t < \tau
]
→ Use System 1 (fast ranking)

If:
[
F_t ≥ \tau
]
→ Use System 2 (diversity-promoting / long-term strategy)

Now fatigue controls which recommender is active.

This is psychologically grounded and novel.

---

# 4️⃣ What Changes Because of Fatigue?

Without fatigue:

* Model recommends 10 similar viral reels.
* User binge consumes.
* Engagement spike.
* Burnout next week.

With fatigue:

* After 3 similar reels → fatigue rises.
* Model injects novelty.
* Or slows intensity.
* User experiences balanced stimulation.
* Retention curve flattens instead of crashing.

It shifts system behavior from:
“Exploit high-click content”
to
“Maintain sustainable engagement.”

---

# 5️⃣ How Do We Compute Fatigue Practically?

You don’t need psychology lab data.

You can approximate using:

### A. Repetition Score

[
Rep_t = Similarity(i_t, i_{t-1})
]

### B. Category Saturation

Entropy of last K items:

Low entropy → high fatigue

### C. Session Length

Long session → increased fatigue

### D. Time Decay

Breaks reduce fatigue:

[
F_t = F_{t-1} \cdot e^{- \delta \cdot time_gap}
]

All computable from logs.

No external data required.

---

# 6️⃣ How Do You Prove It Works?

You show:

1. Baseline model
2. Fatigue-aware model

Compare:

* Time-to-churn
* Session length variance
* Retention curves
* Long-horizon reward

If fatigue-aware model:

* Reduces binge spikes
* Improves long-term retention

Then fatigue is validated.

---

# 7️⃣ Conceptual Insight

Standard recommender models assume utility accumulates linearly.

But in real humans:

[
Marginal\ Utility \downarrow\ with\ repeated\ exposure
]

Fatigue models diminishing returns.

It turns the system from:

Engagement maximizer
into
Sustainable utility optimizer.

---

# 8️⃣ The Deeper Research Contribution

You are introducing a dynamic internal user state variable that:

* Is not directly observed
* Evolves temporally
* Modulates reward
* Alters policy decisions

That’s cognitively inspired modeling.

It moves RecSys from:
Static preference learning
to
State-aware human modeling.

---

# Final Answer (Short Version)

We need fatigue because engagement is not equal to satisfaction.
Fatigue models diminishing marginal utility and cognitive burden.

We incorporate it:

* As an input feature,
* As a penalty in the reward function,
* Or as a gating signal between System 1 and System 2.

It ensures the recommender optimizes long-term retention instead of short-term dopamine spikes.

---

If you’d like, I can next:

* Write the exact pseudo-code of the fatigue-aware algorithm
* Show how this fits into a Transformer-based recommender
* Or explain how reviewers might criticize the fatigue assumption and how to defend it
