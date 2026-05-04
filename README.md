Recommender Systems, Long-Term Optimization, Reinforcement Learning,
User Fatigue, Cognitive State Modeling, Dynamics of Preference,
Sequential Recommendation

# Introduction

Recommender systems have evolved from early accuracy-focused algorithms
to complex industrial systems across e-commerce, media, and social
platforms. Despite these advances, a tension exists between short-term
business-driven goals and long-term user wellbeing. Most current systems
rely on easily measurable signals like clicks and watch time, which
maximize immediate engagement but often do not explicitly model the
user’s cognitive state.

## The Challenge of Engagement Bias

Standard sequential and deep learning models treat user histories as
ordered sequences to predict the next likely interaction. While
effective at capturing recency effects, these models implicitly assume
that maximizing immediate clicks corresponds to maximizing satisfaction.
This approach neglects psychological factors such as **fatigue** and
**diminishing utility**, where repeated exposure to similar content
reduces its value and this leads to platform disengagement.

## Proposed Solution: DS-HCRS

To address these limitations, this work proposes a framework called
**DS-HCRS** which goes beyond a single static strategy by incorporating:

- **Dual-System Architecture:** Balancing a fast "impulsive" model for
  relevance with a reflective "long-term" optimizer.

- **Dynamics of Preference:** Estimating changes in user preference
  using recurrent models.

- **Cognitive State Modeling:** Estimating user fatigue and
  discontentment using Reinforcement Learning objectives.

These three systems alone allows the algorithm to implement more complex
systems like updating user preference (embeddings) on the go, and cold
start handling for new users and items. By framing recommendation as a
cognitive process rather than just a prediction problem, we aim to
develop systems that are sustainable, adaptive, and better aligned with
long-term user experience.

# Literature Survey

## Real-World Recommender Systems: Constraints and Objectives

Recommender systems have seen significant evolutions across various
domains like e-commerce, media streaming, education, and social
platforms. Early studies in this field were more directed towards raw
recommendations going through algorithmic accuracy; recent industrial
requirements have shaped this around deployment constraints.

Application of Recommender Systems in industrial environments highlights
several new dimensions of challenges: massive user populations,
continuously evolving item catalogs, strict latency requirements, and
business-driven optimization goals. These implementations are built over
several stages like candidate generation, ranking, filtering, and
post-processing components. Each stage suffers from a trade-off between
computational efficiency and predictive performance.

Earlier, most recommender systems heavily relied on measurable metrics
like click-through rate (CTR), watch time, or interaction frequency.
These metrics are easy-to-use because developing a testing and
experimentation system around these metrics is relatively easier and
quicker to deploy. However, these signals are imperfect representations
of user satisfaction on a long-term value due to several noise factors
like fatigue and addiction. Therefore, implementations based on these
metrics maximize short-term measurable engagement rather than user
benefit and welfare.

## Evolution of Recommendation Systems

The foundations of recommender systems originate from collaborative
filtering and content-based recommendation methods. These algorithms
modelled on user preferences using historical interaction patterns.
Matrix factorization techniques introduced latent representations that
enabled scalable personalization and became the dominant paradigm for
many years.

With the rise of deep learning algorithms, recommendation systems
shifted towards neural ranking models capable of learning complex
relationships between users and items. This approach improved predictive
performance but worked around the same objective: estimating the
likelihood of immediate interaction.

Further, the introduction of temporal dynamics into the system gave rise
to sequential recommendation models. Architectures such as recurrent
neural networks (e.g., GRU4Rec) and attention-based transformer models
(e.g., SASRec and subsequent variants) treat user interaction histories
as ordered sequences. These systems can learn recency effects and
temporal dependencies, which can further enhance next-item predictions.

Despite their success, sequential recommenders rely on a key assumption:
observed interactions represent positive utility signals. Since the
objective still remains the same, that is, next-click or next-
interaction prediction; this implicitely assumes that maximizing
immediate engagement corresponds to maximizing user satisfaction.
Consequently, users are modeled as behavioral sequences, ignoring
internal states such as fatigue and boredom.

## Reinforcement Learning for Long-Term Recommendation

Reinforcement learning (RL) mechanisms are well known to be built around
the optimization goal of acquiring long term rewards; which seem to fit
perfectly with the problem of algorithms converging towards a short-term
goal issue. Therefore RL has been introduced as a framework for
recommendation to address these temporal dependencies beyond immediate
prediction. RL formulations treat recommendation as a sequential
decision process in which the system selects actions (recommendations)
to maximize cumulative reward over time.

Early approaches focused on contextual bandits, which balance
exploration and exploitation while optimizing short-term feedback
signals. Recent work extends this framework to long-horizon
optimization, modeling delayed rewards such as retention, session
continuation, or user lifetime value.

RL also enables reward shaping, allowing designers to encode multiple
objectives into a single optimization signal. However, existing RL
recommender systems inherit a critical limitation: reward definitions
remain engagement-centric. When modeling long-term outcomes, rewards are
often derived from interaction frequency or platform usage duration. As
a result, RL improves temporal awareness without fundamentally
redefining the optimization objective.

Studies also introduce addiction-aware recommendations, which further
highlight this issue, showing that maximizing engagement-based rewards
can unintentionally encourage compulsive consumption patterns. Thus, RL
provides a mechanism for long-term optimization but does not guarantee
alignment with user wellbeing.

## Generative and LLM-Based Recommender Systems

The recent advances in large language models (LLMs) have started a major
a paradigm shift in most data science fields, including recommender
systems. The use of LLMs in recommender systems introduces generative
and reasoning-based recommendation, while moving away from
discriminative ranking. Generative recommendation frameworks take
recommendations as a sequence generation or reasoning task rather than a
scoring problem.

LLM-based approaches offer a deeper insight about the user preferences
and natural language interaction. This approach also offers improved
cold-start handling through pretrained knowledge representations.
Agent-based recommender architectures further extend this idea by
allowing reasoning loops, planning behaviors, and iterative refinement
of recommendations. Explainable recommendation systems also benefit from
LLMs’ ability to produce human-interpretable justifications.

Despite these advances, the primary optimization objective remains
unchanged. LLMs enhance how recommendations are generated but rarely
reconsider what should be optimized. Most systems still aim to maximize
predicted relevance or engagement likelihood.

## Cognitive and Agentic Recommendation Frameworks

Emerging researches explore recommender systems capable of reflection,
memory, and reasoning. System-2 recommender frameworks try to separate
impulsive engagement from reflective utility by modeling
decision-capable processes.

Agent-based architectures introduce persistent memory structures,
reasoning cycles, and iterative refinement mechanisms. These frameworks
incorporates reasoning, reflection, and refinement attempt to simulate
decision-making processes beyond simple ranking.

These approaches represent an important conceptual shift: recommendation
is framed as a cognitive process rather than a prediction problem.
Memory-aware agents and reasoning loops allow systems to reconsider
their previous decisions and incorporate broader understanding.

However, existing cognitive recommender research primarily focuses on
reasoning capability, rather than human state modeling. While systems
have become more reflective internally, they rarely model user fatigue,
burden, or diminishing utility explicitly.

## Human-Centered Recommendation and Welfare Optimization

Some researches have questioned whether engagement-maximizing goals
align with human welfare. Studies demonstrate that optimizing for
interaction metrics can amplify biases, and encourage excessive
consumption behaviors.

Works on recommendation and temptation explores how exposure to
engagement oriented recommendations can influence a users self-control
and decision making. Similarly, research on welfare-aware optimization
investigates trade-offs between platform performance and user wellbeing;
highlighting the tension between short-term engagement and long-term
satisfaction.

Recommender models which work with addiction awareness, incorporates
risk-sensitive objectives that penalize harmful interaction patterns.
Studies on bias and fairness further reveal that recommendation
embeddings can encode sensitive associations (often personal).

Collectively, these works argue that recommender systems should be
evaluated not only by accuracy but also by their long-term societal and
psychological effects.

## Evaluation Challenges and Control Mechanisms

Depending on what aspects a model is evaluated on, it heavily shapes the
internal designing of that system. Traditional metrics such as
*N**D**C**G*, *R**e**c**a**l**l*@*K*, and
*M**e**a**n**R**e**c**i**p**r**o**c**a**l**R**a**n**k* measure ranking
accuracy under static assumptions. While effective for benchmarking
predictive performance, these metrics fail to capture long-term effects.

Recent research highlights inconsistencies in evaluation protocols,
particularly for explainability and reasoning methods. Offline
evaluation often fails to predict online performance due to feedback
loops and distribution shifts introduced due to deployed circumstances.

Approaches built around control, like conformal risk control, attempt to
mitigate unwanted recommendations, by introducing uncertainty measures.
Studies comparing offline and online evaluation further emphasize that
optimizing standard metrics can reinforce short-term optimization
biases.

## Literature Synthesis and Identified Research Gap

Across the surveyed literature, recommender system research has
progressed along multiple independent trajectories: improved sequential
modeling, long-term optimization via reinforcement learning,
reasoning-enabled recommendation through LLMs, and growing awareness of
human welfare implications. However, these advances remain largely
disconnected.
Table <a href="#tab:collective-findings" data-reference-type="ref"
data-reference="tab:collective-findings">[tab:collective-findings]</a>
highlights the gap between existing capabilities and the components that
are still missing for human-centered recommendation systems.

|l||Y||Y| Capability & Existing Work & Missing Component  
Sequential modeling & Accurate behavioral prediction & Cognitive state
awareness  
RL optimization & Long-term interaction modeling & Human-aligned reward
formulation  
LLM reasoning & Semantic understanding and reflection & Adaptive
decision control  
Welfare research & Identification of harms & Operational system
integration  
Evaluation research & Metric critique and control methods & Alignment
with wellbeing objectives  

This reveals some patterns like: the individual components necessary for
human-centered recommendation exist, yet no unified framework integrates
cognitive modeling, adaptive reasoning, fatigue awareness, and long-term
optimization, all put into a single recommender architecture.

This gap motivates the proposed Dual-System Human-Centric Recommender
with Fatigue-Aware Long-Term Optimization (DS-HCRS), which aims to
bridge behavioral prediction, cognitive modeling, and long-term welfare
optimization within a unified framework.

# Proposed Methodology: DS-HCRS Framework

## Overall System Architecture

The final system is designed to model not just what a user likes, but
also how their behavior changes over time. Instead of treating
recommendation as a static prediction problem, we try to capture three
important aspects:

- What the user prefers

- How their preference changes

- How their cognitive state evolves

To achieve this, the system is divided into three main modules, along
with a few supporting components.

We start with the **KuaiRec dataset**, which acts as our base dataset
(T0, time=0). One important observation from this dataset is that user
preference does not change over time. Because of this, we can safely
train a model to learn stable user and item embeddings.

The first main module is **System 1 (S1)**. This is a Two-Tower
recommender system. Since S1 is trained on the static dataset, it gives
us a good starting point for user preferences (T0 embeddings). Item
embeddings remain fixed, but user embeddings will change later as the
system observes more interactions.

To simulate realistic user behavior, we introduce a **UserModel**
component. This is not a neural network, but a behavioral model. It can
imitate three user begaviors:

- Change in user preference over time

- Probability of clicking based on relevance

- Cognitive states: fatigue and discontentment

When a user sees too many relevant items, fatigue increases. When too
many irrelevant items are shown, discontentment increases.

Using this UserModel, we create a synthetic dataset called
**UserModelDataset**. Here, we start with T0 user embeddings from S1 and
expose the user to many items. This dataset is mainly used to train the
preference modeling component.

The second main module is **System 2 Preference Model (S2P)**. This
module learns how user preferences evolve. It takes a sequence of
interactions, which include item embeddings and click signals, and
encodes them using an LSTM. This helps us capture dynamic preferences.

The third main module is **System 2 Cognitive State Model (S2F)**. This
is responsible for modeling fatigue and discontentment. It is
implemented using **Proximal Policy Optimization (PPO)**. Over time, it
learns how to balance relevant and irrelevant items. This helps control
both fatigue and discontentment.

In addition to these main modules, we have a few supporting components.
The **Index Search module** uses **FAISS IndexFlatIP** to efficiently
retrieve top-k items based on user embeddings. It can work with
embeddings from S1 or S2P, and is also used inside S2F to generate
candidate items.

Another important component is the **User Embedding Updater**. This
module helps keep updating the learnt user embeddings to stay up to date
with the change in preference.

Overall, the architecture works as a pipeline. We start with static
embeddings from S1, then update them using S2P as the user interacts.
Finally, S2F controls what items should be shown by considering fatigue
and discontentment. This allows the system to balance short-term
relevance with long-term user experience in a more realistic way.

## Dataset and Preprocessing

The foundation of our system is the **KuaiRec dataset**. This dataset
contains a rich set of user-item interactions along with multiple
features for both users and items. While this dataset provides many
important assets required towards building a good recommender system, we
only utilise the **watch\_ratio**, which is the ratio of time spent on
an item by the total length of that item.

One key observation we made while working with this dataset is that
**user preferences do not change over time**. In other words, there is
no strong relationship between when an interaction happened and what the
user prefers. Because of this, we treat KuaiRec as a **T0 dataset**,
meaning it represents a *static snapshot* of user preferences.

Since preferences are static here, this dataset is only used to train
our **System 1 (S1)** model to learn stable embeddings for users and
items.

## System 1: Fast Engagement Recommender

System 1 is our base recommender. It learns the user’s initial
preference and the item representation in a shared embedding space. In
our implementation, both the user and item towers use the same idea: the
raw ID and the side features are mapped into a dense vector space, and
then the final representation is used for scoring. The model also learns
a small bias term for each user and item. This is done inside the
`AdaptiveRepresentation` module, which combines an ID embedding with a
feature-based embedding using a learned weight *α*.

For a user *u* and an item *i*, the two towers produce embeddings
*e*<sub>*u*</sub> and *e*<sub>*i*</sub>. Both vectors are normalized
before scoring. The relevance score is then computed using dot product:
$$\begin{equation}
    s(u,i) = \frac{e\_u^\top e\_i}{|e\_u||e\_i|}
\end{equation}$$

In our code, this is followed by optional bias terms:
*s*(*u*, *i*) = *e*<sub>*u*</sub><sup>⊤</sup>*e*<sub>*i*</sub> + *b*<sub>*u*</sub> + *b*<sub>*i*</sub>

So, if the score is high, the item is considered relevant for that user;
otherwise, it is treated as irrelevant. This matches the way the
two-tower model is used in training and inference.

The training data is built in a pairwise way. For each user, the dataset
samples one positive interaction and several negative items. In the
code, positive items are chosen from ratings above a threshold, while
negative items are sampled from lower-rated ones. The model is then
trained to give the positive item a higher score than the negative ones.
The ranking loss is:
ℒ<sub>*r**a**n**k*</sub> = −log *σ*(*s*(*u*, *i*<sup>+</sup>) − *s*(*u*, *i*<sup>−</sup>))

When there are multiple negatives, the loss is summed over all of them.
This makes the model learn a clean separation between relevant and
irrelevant items in the embedding space.

We train System 1 in two phases. First, the feature-based embeddings are
trained, which gives us a strong representation of the entity only using
the features. This gives us feature-based embeddings as user embeddings
*e*<sub>*u**f*</sub> and item embeddings *e*<sub>*i**f*</sub>. Later
both the feature-based embeddings and ID-based embeddings are used
together in training while feature-based embeddings are kept frozen.
This allows the embeddings to learn to use ID-based embeddings only as a
corrector or offset embeddings. This gives us ID-based embeddings user
embeddings *e*<sub>*u**i*</sub> and item embeddings
*e*<sub>*i**i*</sub>. The final embeddings are calculated as:
*e*<sub>*u*</sub> = *e*<sub>*u**f*</sub> + *e*<sub>*u**i*</sub>
*e*<sub>*i*</sub> = *e*<sub>*i**f*</sub> + *e*<sub>*i**i*</sub>

This gives us a good starting representation for users and items before
any dynamic behavior is introduced. That means it learns a static
preference state, which we treat as the user’s (T\_0) embedding. In
other words, S1 tells us what the user generally likes at the start,
before later modules update that preference over time.

### Cold Start Handling

A nice thing about S1 is that it does not depend only on IDs. In the
`AdaptiveRepresentation` module, each user and item is represented using
both an **ID embedding** and a **feature-based embedding**. The model is
first trained on the feature side, so it learns a strong representation
from the available user and item features. Meanwhile, during the
training of ID-based embeddings, since we use features as well while
keeping them frozen, the model learns ID-based embeddings as an offset
pr correction representation. Therefore, a new user or a new item can
still be mapped into the same latent space using only its features.

This is how we handle cold start in practice. For a new user, we only
need the user features already available in KuaiRec. For a new item, we
only need the item-side features, such as its metadata and
category-related information. Once those features are passed through the
model, we get an embedding that can already be used for recommendation.
The ID embedding part can start from a random initialization with a very
small magnitude and then get updated later as more interactions are
collected.

So, the main idea is simple: **features give us a usable starting point,
and IDs help refine the representation over time**. This is why S1 works
well as the base model for both old and new entities, and why it is also
a good foundation for the later dynamic modules.

## User Behavior Simulation (UserModel)

This component is one of the strongest parts of the whole system,
because it connects the theory of user behavior with the actual training
data. Instead of assuming that users always behave in a clean and fixed
way, we simulate how a user may react over time when they see relevant
items, irrelevant items, repeated items, or too much variety. This is
what gives the later models realistic supervision.

At its core, the **UserModel** keeps track of a user’s **preference
vector**, **fatigue** and **discontentment**. The user also has a
probabilistic click behavior. For an item *i*<sub>*t*</sub> at time *t*,
we first compute a relevance score *r*<sub>*t*</sub> from the current
user preference *u*<sub>*t*</sub> and item embedding *v*<sub>*t*</sub>.
In our implementation, relevance is based on the normalized dot product
between user and item vectors. A simple form is:
$$\begin{equation}
    r\_t = \frac{u\_t^\top v\_t}{|u\_t| , |v\_t|}
\end{equation}$$

Then the click is sampled probabilistically:
*c*<sub>*t*</sub> ∼ Bernoulli(*p*<sub>*t*</sub>)

where the click probability is reduced when fatigue is high. In code,
this is implemented by blending the relevance score toward zero as
fatigue grows:
*p*<sub>*t*</sub> = (1 − *f*<sub>*t*</sub>), *r*<sub>*t*</sub>

So if the user is fresh, relevant items are likely to be clicked. But if
the user is already fatigued, even good items become less effective.

The user preference is also updated when a click happens. This is the
part that models **preference drift**. The preference vector slowly
moves toward the clicked item embedding.
*ũ*<sub>*t* + 1</sub> = (1 − *η*), *u*<sub>*t*</sub> + *η*, *v*<sub>*t*</sub>

It is then passed through a preference update manager that rescales it
so that the new preference vector has the same norm as the other learned
user embeddings.
*u*<sub>*t* + 1</sub> = NormMatch(*ũ*<sub>*t* + 1</sub>)

This keeps the new preference vector stable and compatible with the
embedding space used by the recommender.

The model also maintains two separate cognitive states: **fatigue** and
**discontentment**. These are updated differently depending on whether
the shown item is relevant or not. If the item is relevant, fatigue
increases because the user is being shown many good items in a row, and
discontentment decreases because the experience is still pleasant. If
the item is irrelevant, fatigue decreases a bit, but discontentment
increases because the user is seeing items that do not match their
interest.
*f*<sub>*t* + 1</sub> = clip(*f*<sub>*t*</sub> + **1**\[*r*<sub>*t*</sub> &gt; *τ*\]*η*<sub>*f*</sub><sup>+</sup> − **1**\[*r*<sub>*t*</sub> ≤ *τ*\]*η*<sub>*f*</sub><sup>−</sup>)
*d*<sub>*t* + 1</sub> = clip(*d*<sub>*t*</sub> − **1**\[*r*<sub>*t*</sub> &gt; *τ*\]*η*<sub>*d*</sub><sup>+</sup> + **1**\[*r*<sub>*t*</sub> ≤ *τ*\]*η*<sub>*d*</sub><sup>−</sup>)

where *τ* is the relevance threshold, *f*<sub>*t*</sub> is fatigue, and
*d*<sub>*t*</sub> is discontentment.

These two states behave differently in the session. **Fatigue** leads to
a temporary exit. If fatigue becomes too high, the user stops
interacting for the rest of the day. The next day, the fatigue level is
reduced using a cooldown factor, so the user can come back in a partly
recovered state. On the other hand, **discontentment** is more severe.
If discontentment reaches the maximum level, the user is considered to
have left the platform for good.

Overall, this UserModel acts like a bridge between the clean training
setup and real human behavior. It generates interaction sequences where
preference can drift, clicks are probabilistic, fatigue can build up,
and discontentment can eventually end the session. That makes the
synthetic data much more useful for training the later System 2 modules,
especially the preference model and the feedback model.

## Synthetic UserModelDataset

To train the preference modeling part of the system, we generate a
synthetic dataset using the UserModel. This dataset is built on top of
the embeddings learned from System 1.

This process starts by sampling **T0 user embeddings** from S1. Each
sampled user is then assigned a UserModel instance, which acts like a
simulated user. Then this user is exposed to a sequence of items and how
they interact over time is recorded. This creates realistic interaction
histories consisting of item features, embeddings, and click signals.

A key design choice here is that **fatigue and discontentment are
disabled** during this process. This is done intentionally to focus on
how the user’s **preference changes over time** without any external
effects interfering with it.

The item sequence is generated in a mixed way. Some items are relevant
(close to the user’s preference), while others are random. This ensures
that the user sees a variety of items, and their preference evolves
naturally based on what they interact with. It helps us train the S2P
module to understand how user preferences shift as more interactions
happen.

## System 2 - Preference Component

The preference component (S2P) is the part of the system that learns how
user preference changes over time. It takes the initial user embedding
from S1 and updates it using the user;s interaction history. So instead
of treating the user as a fixed point, this module treats preference as
something that evolves step by step.

The input to this model is a sequence of interactions. For each step, we
use the item embedding together with the click signal. The user
embedding from S1 is also used as the starting representation, so the
model knows where the user began before the history is applied. This
setup is exactly what allows the model to work with variable-length
sessions.

We can write the input history as:
*H*<sub>*t*</sub> = (*e*<sub>1</sub>, *b*<sub>1</sub>, *c*<sub>1</sub>), (*e*<sub>2</sub>, *b*<sub>2</sub>, *c*<sub>2</sub>), …, (*e*<sub>*t*</sub>, *b*<sub>*t*</sub>, *c*<sub>*t*</sub>)

where *e*<sub>*j*</sub> is the embedding and *b*<sub>*j*</sub> is the
bias of the *j*-th item shown to the user, and *c*<sub>*j*</sub> is the
click indicator. The history encoder processes this sequence along with
the initial user embedding *u*<sub>0</sub>, and produces an updated
preference embedding:
*u*<sub>*t*</sub> = S2P(*u*<sub>0</sub>, *H*<sub>*t*</sub>)

The important thing here is that the output embedding lives in the
**same latent space** as the user and item embeddings learned by S1.
That means the result is not some separate hidden state that cannot be
used later. It is a proper **updated user preference embedding**, which
can directly interact with item embeddings through dot product scoring.

This is why S2P is the temporal preference learning module. It gives us
the transition from *T*<sub>0</sub> to *T*<sub>*n*</sub>: the user
embedding starts from the static base learned by S1 and then gets
refined as more interactions arrive. So after enough history, the system
has a much better picture of what the user currently prefers, not just
what they liked at the beginning.

In the implementation, the item tower and user tower from S1 are frozen
while this module is trained. That is a useful design choice, because it
keeps the base embedding space stable and lets S2P focus only on
learning the change in preference from history. So the role of S2P is
not to relearn everything from scratch, but to update the user
representation in a controlled and time-aware way.

## System 2 - Cognitive State Component

The cognitive state component (S2F) is one of the most advanced parts of
the system. It is the module that learns how to balance **good
recommendations**, **fatigue**, and **discontentment** using
reinforcement learning. This implementation uses **Proximal Policy
Optimization (PPO)** as the RL algorithm.

The **state** given to PPO is built from the updated user preference
embedding coming from S2P, plus a few session-level signals. So the
observation at time *t* is:

$$\begin{equation}
    o\_t = \begin{bmatrix}
            p\_t, b\_t \\
            \text{recent\\click\\rate}\_t \\
            \text{similarity\\of\\recent\\items}\_t \\
            \text{diversity\\score}\_t \\
            \text{session\\length}\_t
           \end{bmatrix}
\end{equation}$$

where *p*<sub>*t*</sub> is the current preference embedding and
*b*<sub>*t*</sub> is the preference bias. In the code, the observation
space is explicitly defined as a vector of size
*d*<sub>embed</sub> + 1 + 4, so the embedding part stays in the same
latent space as S1 and S2P, and the four extra values summarize the
current user state.

The extra state signals are computed from recent interaction history.
The **recent click rate** measures how often the user clicked in the
last few steps. The **similarity of recent items** is the mean
cosine-like similarity between consecutive recent item embeddings. The
**diversity score** is simply 1 − similarity\_of\_recent\_items. and
**session length** counts how many interactions have happened since the
last temporary exit caused by fatigue. These values help PPO understand
whether the user is still comfortable, getting bored, or starting to
disengage.

As for the **action space**, PPO does not choose directly from the whole
catalog. Instead, it chooses from a **mixed candidate pool of 100
items**, made of:

- 70 relevant items

- 20 semi-relevant items

- 10 random items

First, the current preference embedding from S2P is passed through ANN
search using FAISS. That gives relevant candidates. Then a smaller group
of semi-relevant items is sampled the in a similar way. Finally, random
items are added to introduce variety. PPO then selects one action from
this fixed pool:
*a*<sub>*t*</sub> ∈ 0, 1, …, 99

and that action is mapped to one of the 100 candidate items. This design
does not require the agent to search over the full item universe, thus
preventing the use of an enoumous action space while also staying
compatible with cold start in case new items are added. It only needs to
learn how to balance relevance and exploration inside a controlled set.

Once the chosen item is shown to the simulated user, the environment
updates the user state. If the item is relevant, the user is more likely
to click and the preference vector is updated through the user
simulator. If the user becomes fatigued, the environment forces a
temporary break for the rest of the day. If discontentment becomes too
high, the user leaves permanently. So S2F is not just predicting clicks;
it is directly learning how recommendation choices affect long-term user
behavior.

The reward function is designed to reflect that. In code, it strongly
rewards clicks, but penalizes temporary fatigue, permanent exit, and
high internal fatigue or discontentment levels:
$$\begin{equation}
    r\_t = 
        \sum
        \begin{bmatrix}
            2\cdot \mathbf{1}\[\text{click}\] \\
            - 8\cdot \mathbf{1}\[\text{cannot\\interact}\] \\
            - 100\cdot \mathbf{1}\[\text{gone\\for\\good}\] \\
            - 2 f\_t^{1/4} \\
            - 2 d\_t^{1/4}
        \end{bmatrix}
\end{equation}$$

where *f*<sub>*t*</sub> is fatigue and *d*<sub>*t*</sub> is
discontentment. So the policy is encouraged to keep the user engaged,
but not by overusing either relevant items or irrelevant items. That is
exactly how S2F balances the two failure modes from the simulator.

For PPO itself, the policy is trained with the usual clipped objective:
ℒ<sup>PPO</sup>(*θ*) = 𝔼<sub>*t*</sub>\[min (*ρ*<sub>*t*</sub>(*θ*)*A*<sub>*t*</sub>, clip(*ρ*<sub>*t*</sub>(*θ*), 1 − *ϵ*, 1 + *ϵ*)*A*<sub>*t*</sub>)\]

where *ρ*<sub>*t*</sub>(*θ*) is the policy ratio and *A*<sub>*t*</sub>
is the advantage estimate. In the training setup, PPO uses a discount
factor of 0.997, an entropy coefficient of 0.05, and a small MLP policy
network with 82977 parameters.

## Candidate Generation and Retrieval

To efficiently retrieve items from a large catalog, we use
**IndexFlatIP** method. This index performs **inner product search**,
meaning it retrieves items whose embeddings have the highest dot product
with a given query embedding.

Since all our user and item representations lie in the same embedding
space, this becomes a very natural choice. Given a user embedding *u*,
the index simply returns items *i* that maximize:
*u*<sup>⊤</sup>*v*<sub>*i*</sub>

This makes retrieval very fast and scalable, while still being aligned
with how relevance is defined in S1.

This index is used across multiple parts of the system:

- In **S1**, it is used for fast top-k retrieval based on static user
  embeddings

- In **S2P**, it is used with updated preference embeddings to get more
  context-aware recommendations

- In **S2F**, it is used twice: first to generate a pool of candidate
  items (relevant and semi-relevant), and later to retrieve final
  recommendations

## User Embedding Update Mechanism

This mechanism is a small part of the code, but it plays a very
important role in the system. Its main idea is simple: after S2P learns
a better representation of the user from session history, that updated
embedding can be written back into S1. In this way, the system does not
need to rebuild the user state from scratch every time. Instead, the
current session history gets condensed into a single updated user
embedding.

If we write the S1 user embedding as *u*<sup>(0)</sup> and the updated
preference embedding from S2P as *u*<sup>(*t*)</sup>, then the update
step can be seen as:
*u*<sup>(0)</sup> ← *u*<sup>(*t*)</sup>

with the corresponding bias updated in the same way. In the
implementation, the update is done carefully so that the stored S1
embedding is adjusted while still staying compatible with the
feature-based part of the representation. So the system does not just
overwrite the vector blindly; it updates the internal embedding in a way
that preserves the learned latent space.

The purpose of this mechanism is mostly practical. Since S2P is based on
an LSTM and works over interaction history, using it too often can
become expensive as the history grows. By saving the latest updated user
embedding back into S1 and clearing the current session history, we keep
the user state compact. This makes it possible to rely on S1-style
retrieval again for a while, without needing to query S2P every single
time.

So this mechanism acts like a memory compression step. It takes the long
interaction history, turns it into one updated embedding, and resets the
session state. This helps keep the system efficient while still allowing
user preferences to evolve over time. In a way, it creates a balance
between fast retrieval and deeper temporal modeling.

# Evaluation Design

Evaluation is done in a simulation loop over sampled users and time
steps. At each step, the model produces recommendations, the simulated
user reacts to the top item, and the metrics are accumulated. This way
allows observing how a recommender performs over time, where preference
keeps changing and the user can get fatigued or discontented.

Let *R*<sub>*u*</sub><sup>*k*</sup> be the top-*k* recommended items for
user *u*, and let *T*<sub>*u*</sub> be the set of true relevant items
for that user.

## Core Recommendation Accuracy

**Hit Rate @ K**  
Checks whether at least one relevant item appears in the top-*k* list.

$$\begin{equation}
    \mathrm{HR@K}=\frac{1}{|U|}\sum\_{u\in U}\mathbf{1}\left(|R\_u^k\cap T\_u|&gt;0\right)
\end{equation}$$

**Precision @ K**  
Measures the fraction of the returned top-*k* items that are relevant,
using the deduplicated top-*k* set.

$$\begin{equation}
    \mathrm{Precision@K}=\frac{1}{|U|}\sum\_{u\in U}\frac{|R\_u^k\cap T\_u|}{|R\_u^k|}
\end{equation}$$

**Recall @ K**  
Measures how much of the relevant set is recovered in the top-*k* list.

$$\begin{equation}
    \mathrm{Recall@K}=\frac{1}{|U|}\sum\_{u\in U}\frac{|R\_u^k\cap T\_u|}{\min(k,;|T\_u|)}
\end{equation}$$

**NDCG @ K**  
Gives higher credit to relevant items appearing earlier in the ranking.

$$\begin{equation}
    \mathrm{DCG@K}=\sum\_{i=1}^{|R\_u^k|}\frac{\mathbf{1}(r\_i\in T\_u)}{\log\_2(i+1)}
\end{equation}$$

$$\begin{equation}
    \mathrm{NDCG@K}=\frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}
\end{equation}$$

where IDCG@K is the best possible DCG for that user.

**MRR @ K**  
Measures how early the first relevant item appears in the ranked list.

$$\begin{equation}
    \mathrm{MRR@K}=\frac{1}{|U|}\sum\_{u\in U}\frac{1}{\mathrm{rank}\_u}
\end{equation}$$

where rank<sub>*u*</sub> is the position of the first relevant item in
the top-*k* list, and 0 if none is present.

## Long-Term Engagement and User Welfare

**Session Depth**  
Counts how many steps the user stays active in a session without
becoming fatigued.

$$\begin{equation}
    \mathrm{SessionDepth}\_u=\sum\_{t=1}^{T}\mathbf{1}(\text{user not fatigued at } t)
\end{equation}$$

**Return Rate**  
Measures how often a fatigued user comes back after a temporary break.

$$\begin{equation}
    \mathrm{ReturnRate}\_u=\sum\_{t=1}^{T}\mathbf{1}(\text{user returns after fatigue})
\end{equation}$$

**Time-to-Boredom**  
Measures the average number of steps before fatigue interrupts the
session.

$$\begin{equation}
    \mathrm{TimeToBoredom}\_u=\frac{1}{m}\sum\_{j=1}^{m} \ell\_j
\end{equation}$$

where *ℓ*<sub>*j*</sub> is the length of the *j*-th active stretch.

**Time-to-Discontent**  
Measures the average number of steps before discontentment stops
interaction.

$$\begin{equation}
    \mathrm{TimeToDiscontent}\_u=\frac{1}{n}\sum\_{j=1}^{n} d\_j
\end{equation}$$

where *d*<sub>*j*</sub> is the length of the *j*-th uninterrupted period
before discontentment appears.

**Acceptance Rate**  
Measures how often the shown recommendations are clicked.

$$\begin{equation}
    \mathrm{AcceptanceRate}\_u=\frac{1}{T}\sum\_{t=1}^{T}\mathbf{1}(\text{click}\_t)
\end{equation}$$

## Fatigue and Boredom Metrics

**Content Repetition Rate**  
Measures how similar consecutive recommended item embeddings are on
average.

$$\begin{equation}
    \mathrm{RepetitionRate}\_u=\frac{1}{T-1}\sum\_{t=1}^{T-1}\cos(e\_t,e\_{t+1})
\end{equation}$$

**Novelty**  
Measures how different recent recommendations are from each other.

$$\begin{equation}
    \mathrm{Novelty}\_u=\frac{1}{T-1}\sum\_{t=1}^{T-1}\left(1-\cos(e\_t,e\_T)\right)
\end{equation}$$

**Intra-List Diversity**  
Measures how diverse the recent recommendation window is by looking at
pairwise similarity inside that window.

$$\begin{equation}
    \mathrm{ILD}\_u=1-\frac{1}{m(m-1)}\sum\_{i\neq j}\cos(e\_i,e\_j)
\end{equation}$$

**Fatigue Score**  
A simple composite signal built from boredom, novelty, and diversity.

FatigueScore<sub>*u*</sub> = TimeToBoredom<sub>*u*</sub> − Novelty<sub>*u*</sub> − ILD<sub>*u*</sub>

## Human-Centered Recommendation Metrics

**Diversity**  
Measures how diverse the recommendations are by looking at every two
consequtive recommendations.

Diversity<sub>*u*</sub> = ILD<sub>*u*</sub>

**Serendipity**  
Combines novelty with repetition-like behavior in the implemented code.

Serendipity<sub>*u*</sub> = Novelty<sub>*u*</sub> + RepetitionRate<sub>*u*</sub>

**Coverage**  
Measures how much of the true-item set is touched by the recommendations
over the evaluation run.

$$\begin{equation}
    \mathrm{Coverage}\_u=\frac{|\mathcal{R}\_u|}{|\mathcal{T}\_u|}
\end{equation}$$

where ℛ<sub>*u*</sub> is the set of unique recommended items and
𝒯<sub>*u*</sub> is the set of unique true items collected during
evaluation.

## Baselines and Ablation Study

To properly evaluate the proposed system, we compare it against several
baseline models. These baselines are chosen to represent different
levels of complexity, from classical recommendation methods to
components of our own system.

We include the following models:

- **Alternating Least Squares (ALS)**  
  A classical collaborative filtering method based on matrix
  factorization. It learns user and item latent factors by minimizing
  reconstruction error on observed interactions. This serves as a strong
  traditional baseline.

- **Bayesian Personalized Ranking (BPR)**  
  A pairwise ranking-based approach that learns to rank observed
  interactions higher than unobserved ones. It is widely used for
  implicit feedback settings and provides a good comparison for ranking
  quality.

- **T0 Embeddings of System 1 (S1)**  
  This is our two-tower model trained on the T0 dataset. It represents a
  strong neural baseline that captures user preference without any
  temporal dynamics.

- **System 2 Preference Model (S2P)**  
  This model uses interaction history to update user embeddings. It
  captures preference drift but does not consider fatigue or
  discontentment. This baseline isolates the benefit of temporal
  modeling alone.

- **System 2 Cognitive Model (S2F)**  
  This is the full reinforcement learning-based model. It incorporates
  both preference evolution and cognitive states such as fatigue and
  discontentment. This represents the most complete version of our
  system.

# Results and Analysis

## Core Recommendation Metrics

Fig. <a href="#fig:core_metrics" data-reference-type="ref"
data-reference="fig:core_metrics">1</a> shows the performance of
different models across multiple interaction stages (T0, T100, T300,
T500). These stages represent how the system behaves as more user
interaction history becomes available.

<figure id="fig:core_metrics" data-latex-placement="!h">
<img
src="../../attachments/448b9b45434d03cd8cb76b793ab04aa086d3e311.png" />
<figcaption>Performance comparison of all models across different
interaction stages (T0, T100, T300, T500). Static models such as ALS and
S1 perform well at early stages but degrade as user preferences evolve.
In contrast, S2P and S2F improve over time by adapting to interaction
history, with S2F showing a slight trade-off in accuracy due to
cognitive-state-aware recommendations.</figcaption>
</figure>

### Performance at Initial Stage (T0)

At T0, where no interaction history is available, the **S1 model
performs the best across all accuracy and ranking metrics**. This is
expected, since S1 is trained on static user preferences and is
specifically designed for this scenario.

On the other hand, both **S2P and S2F perform relatively poorly at this
stage**. These models rely on interaction history to update user
preferences, and in the absence of such history, they do not have enough
information to perform well.

This shows that **static models are strong in cold-start or early-stage
scenarios**.

### Effect of Preference Dynamics (T100 to T300)

As interaction history increases, a clear shift in performance can be
observed.

The performance of **S1 drops significantly** across all metrics. This
indicates that static embeddings quickly become outdated when user
preferences start evolving.

In contrast, **S2P shows a strong improvement** in all metrics. Since it
updates user embeddings using interaction history, it is able to track
preference changes effectively. As a result, it outperforms all static
and classical models in mid and later stages.

This demonstrates the importance of **modeling preference dynamics
instead of relying on fixed embeddings**.

### Effect of Cognitive Modeling (S2P vs S2F)

When comparing S2P and S2F, an interesting pattern emerges.

S2P achieves **slightly higher values in precision, recall, and NDCG**,
indicating that it is more focused on recommending highly relevant
items.

However, S2F performs slightly lower on these metrics. This is expected,
since S2F does not purely optimize for relevance. Instead, it balances:

- Relevance

- Fatigue

- Discontentment

Because of this, S2F occasionally introduces less relevant or more
diverse items, which slightly reduces short-term accuracy.

This highlights an important trade-off:

**Optimizing only for relevance (S2P) leads to better short-term
accuracy, while incorporating cognitive factors (S2F) leads to more
balanced and sustainable recommendation behavior.**

### Long-Term Behavior (T300 to T500)

At later stages, both S2P and S2F achieve very high performance, with:

- Hit Rate approaching 1

- MRR close to 1

This means that **relevant items are almost always present, and often
appear at the top of the recommendation list**.

Meanwhile, S1 continues to degrade, and classical methods like ALS and
BPR fail to adapt to changing preferences.

This further reinforces that:

**Static and traditional models are not suitable for dynamic
recommendation environments.**

## Simulation Metrics

The second set of results focuses on the simulated behavior of the user,
rather than only the ranking quality of the recommender. These metrics
are important because our system is not just trying to recommend
relevant items. It is also trying to control fatigue, avoid
discontentment, and keep the interaction healthy over time.
Fig. <a href="#fig:simulation_metrics" data-reference-type="ref"
data-reference="fig:simulation_metrics">2</a> shows four key behavioral
metrics: return rate, acceptance rate, time to boredom, and coverage.

<figure id="fig:simulation_metrics" data-latex-placement="!h">
<img
src="../../attachments/4d04100c710875a98f0b7f2248832938cc054ce2.png" />
<figcaption>Comparison of behavioral metrics including return rate,
acceptance rate, time to boredom, and coverage. While relevance-driven
models (ALS, S1, S2P) lead to quicker fatigue, S2F significantly
increases time to boredom by balancing relevance and diversity. The
coverage trends highlight the interaction between recommendation
behavior and evolving user preference within the
simulation.</figcaption>
</figure>

### Behavioral Trade-offs Across Models

A clear pattern appears in the **return rate** and **time to boredom**
plots. The classical models, **ALS** and **S1**, show a relatively high
return rate and a low time to boredom. This suggests that these models
often keep recommending items that are relevant enough to trigger
interaction, but also repetitive enough to make the user tired fairly
quickly. In other words, they are able to engage the user, but they do
not manage the experience very well over time.

**S2P** follows a similar pattern. Even though it updates the user
preference using history, it is still strongly relevance-driven. Because
of that, it can keep showing items that are close to the user’s current
interest, which is useful in the short term but can still lead to
fatigue. This explains why S2P does not completely escape the "high
relevance, quick boredom" behavior.

**S2F** behaves very differently. It has the **highest time to boredom**
by a large margin, which is exactly what we would want from a
cognitive-state-aware model. This means the user stays active for much
longer before reaching fatigue. At the same time, its return rate is
lower than the other strong models, which suggests that it is avoiding
excessive overload and spreading the interaction more carefully. This is
a good sign, because S2F is supposed to balance relevance with cognitive
comfort, not just maximize clicks.

So overall, this plot supports the main idea of the project: **pure
relevance is not enough, and the best long-term system is the one that
knows when to slow down.**

### Acceptance Rate Is Not the Full Story

The **acceptance rate** plot is more subtle.

At first glance, **BPR** looks surprisingly strong here, even though it
performs poorly on the core ranking metrics. This is a good reminder
that acceptance rate alone does not tell the full story. Since it is
averaged over time, it can be influenced by a few early clicks or by
short periods where the model happens to align with the user’s changing
preference. So a high acceptance rate does not necessarily mean the
model is actually good overall.

For **ALS**, **S1**, and **S2P**, the acceptance rate is lower than BPR,
which may seem surprising at first. But this makes sense when we connect
it with the earlier plots. These models are often producing highly
relevant items, but they can also push the user toward fatigue faster.
So the user may still click, but the interaction does not stay healthy
for long.

**S2F** sits in the middle. Its acceptance rate is better than S1 and
S2P, but lower than BPR. This is reasonable, because S2F is not only
trying to get clicks. It is also trying to manage fatigue and
discontentment. That makes it less aggressive, but more balanced.

The important takeaway is that **acceptance rate must be read together
with ranking quality and time-based behavior**. On its own, it is too
easy to misinterpret.

### Coverage and the Exploration Paradox

The **coverage** plot is the most interesting one, and honestly, it
reveals a very important limitation of the simulation setup.

At first, **ALS** and **S1** appear to have the highest coverage.
**S2P** and **S2F** are lower, with S2F being the lowest. At face value,
this seems strange, because we would normally expect dynamic models to
explore more items. But in this simulator, coverage is not a completely
fixed quantity. The set of "true" items is tied to the user’s evolving
preference, and that preference is shaped by the recommendations the
user sees.

When a model shows a broader variety of items, the user preference can
expand too, which enlarges the set of items considered relevant later
on. So in this setup, a model that explores more can actually end up
with a larger true-item space, which makes coverage look smaller.

This is why the result is not necessarily a bug in the model. It may
actually be a sign that the simulator itself is capturing preference
growth in a very strong way. In other words:

- **ALS and S1** stay in a narrower region of the item space, so the
  true-item set remains smaller.

- **S2P and S2F** expose the user to a wider range of items, so the
  preference space grows, and coverage goes down relative to that larger
  set.

This is a very important point, because it means the coverage metric
here is not just measuring "how much of the catalog was recommended." It
is also measuring "how much the user’s relevant space expanded during
simulation." That makes it useful, but also harder to interpret.

A second reason this effect may be happening is that the simulation
currently always takes the **first item from the top-k list** to
continue the next step. That makes the trajectory more deterministic and
may limit the exploration benefit of the broader top-k list. So the low
coverage of S2P and S2F may partly come from the simulation policy
itself, not only from the recommender.

### What We Learned from the Dropped Metrics

Some of the other metrics did not contribute much extra information, so
they were not emphasized in the final analysis.

The **content repetition rate**, **time to discontent**, and
**serendipity** plots were almost flat across models, so they did not
help distinguish the systems in a meaningful way. Likewise, the
**fatigue score** was heavily dominated by the time-to-boredom term, so
it behaved very similarly to that metric and did not add much new
insight.

That is actually useful to know, because it tells us that not every
metric is equally informative in this simulator. Some of them are
redundant, while others are too sensitive to the current simulation
design. So for the final discussion, the most meaningful behavioral
signals are the ones that clearly separate the models: **return rate,
acceptance rate, time to boredom, and coverage**.

## Summary of Key Observations

The results highlight a few key points:

1.  **Static models work well only at the start.**  
    They give good recommendations early, but struggle as user
    preferences change.

2.  **S2P improves performance over time.**  
    It tracks changing preferences, but still focuses mainly on
    relevance, which can lead to user fatigue.

3.  **S2F offers a better balance.**  
    It may slightly reduce short-term accuracy, but it delays boredom
    and supports longer, more stable interactions.

4.  **Classical models and S1 have weak long-term behavior.**  
    They keep users engaged only for a short time before fatigue builds
    up.

5.  **Evaluation results should be interpreted carefully.**  
    The simulated user changes based on the model’s recommendations, so
    some metrics are influenced by the model itself.

Overall, these results suggest that recommendation systems should move
beyond static predictions and consider both changing preferences and
user behavior.

# Future Scopes

This project already shows that static recommendation is not enough when
user preference keeps changing. At the same time, the results also
showed a few clear limits in the current design. The future work of this
system is therefore not about adding random new ideas. It is about
fixing the missing pieces that matter most.

## Gating Mechanism

The biggest missing module is the gating mechanism.

Right now, the system has three useful modes:

- **S1** for fast static retrieval

- **S2P** for updated preference learning

- **S2F** for cognitive-state-aware recommendation

All three are useful, but they can not be used all the time. S1 based
embeddings quickly lose relevance as preference updates, and S2P is a
very expensive module. The real challenge is to decide **when** to use
each one. That is what the gating mechanism should do. A good gate
should balance all our previous goals, along with the addition of
computational cost.

This is important because S2F is the strongest module, but it should not
always be the default choice. The system becomes much more practical
only when it learns to switch between the simpler and heavier modules in
a controlled way.

## Simulation and Evaluation Limitations

The simulation helped us test the model, but it also exposed some issues
in the evaluation setup.

### Top-K selection bias

In the current simulation, the next step is often driven by the first
item from the top-k list. This makes the interaction path too
deterministic. A random or probabilistic choice from the top-k set would
give a more realistic simulation.

### Metric design issues

Some metrics turned out to be weak, redundant, or hard to interpret. A
few of them gave almost the same pattern, so they did not add much new
information. Coverage was also tricky, because the size of the true-item
set changes as the user preference changes during simulation. So
coverage here is not just a measure of recommendation breadth. It is
also affected by how the user state grows over time.

This shows that we need better metrics that can separate:

- exploration

- preference growth

- and actual recommendation quality

### Missing user-level evaluation

Most of the current evaluation is averaged over users. That is useful,
but it hides individual failures. A model can look good on average even
if some users fail early.

A better future evaluation should include measures like:

- success rate per user

- how many users survive the full horizon

- and how many users become fatigued early

## UserModel Limitations

The UserModel is useful, but it is still a handcrafted simulator.

### Intuition-driven behavior

The current behavior rules were designed manually. They are reasonable,
but they are not based on a formal psychological study. So the model is
practical, but it is still intuitive.

### Tuned constants

Several values were chosen by experiment, including:

- preference volatility

- fatigue update weights

- and discontentment update weights

These constants work for the current setup, but they are still
hand-tuned. That means the behavior is tied to our design choices, not
learned from real user data.

### Single-drift preference dynamics

At the moment, preference mostly drifts toward items the user clicks.
This works well for one kind of behavior, but not for users whose
interests change in sharper or more complex ways.

### Multi-state preference dynamics

A more realistic user may move between multiple preference states over
time. For example, a user can shift between different moods, topics, or
interests. A future simulator could sample several preference anchors
and switch between them over time.

### Multi-base preference limitation

The current embedding setup still assumes that a user lives in one
coherent region of the space. But in real life, a user can like very
different item groups at the same time.

It may also explain why the T0 scores are not as strong as expected. The
same user embedding is being pulled toward different item groups, and
that makes the final representation harder to fit cleanly.

## Explicit Dislike Modeling

The current system does not track a clear dislike event. In practice,
click and no-click are not the same as like and dislike. This makes the
random sampling pool less controlled than it should be.

A future version should:

- record disliked items explicitly

- avoid showing them again in random sampling

- and keep them out of the candidate pool when possible

This would make the simulated interaction cleaner and more realistic.

## Recommendation Memory

The system does not yet keep a strict memory of what has already been
recommended. So, in theory, the same item can appear again in the
recommendation list.

That did not happen too often in our current runs, but it is still a
limitation. A better system should track recommendation history and
filter duplicates more carefully.

## Better Observation Space for S2F

The current RL observation space is already useful, but it is still
limited. Right now, S2F observes:

- the updated user preference embedding

- recent click rate

- similarity of recent items

- diversity score

- and session length

This is a good start, but it may not be enough for a stronger policy. In
future work, the observation space can include richer behavioral signals
and more context from the session. That would give the algorithm a
better view of how the user is actually reacting.

## Final Note

The main value of these future directions is not that the current system
is incomplete. It is that the current results already point to the next
set of problems very clearly. The most important next step is to make
the system smarter about **when to use each module**, and to make the
simulation and evaluation more faithful to real user behavior.

That is where the next version of this project can become much stronger.

# Conclusion

In this work, we moved beyond traditional recommender systems that focus
only on short-term relevance. We proposed a system that models not just
what a user likes, but also how their preference changes over time and
how their cognitive state evolves during interaction.

The results show that static models perform well only at the beginning,
but fail as user behavior becomes dynamic. Incorporating preference
dynamics through S2P significantly improves performance over time.
Further, the S2F model introduces a more balanced recommendation
strategy by accounting for fatigue and discontentment, leading to better
long-term interaction behavior.

Overall, this work highlights that effective recommendation is not just
about predicting the next click, but about sustaining a healthy and
engaging user experience over time.
