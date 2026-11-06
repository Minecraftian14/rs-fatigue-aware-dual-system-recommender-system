# Dual-System Human-Centric Recommender with Fatigue-Aware Long-Term Optimization (DS-HCRS)

**DS-HCRS** is a dual-system recommender framework that explores how recommendation systems can balance immediate content relevance with long-term user experience. Unlike conventional recommenders that primarily optimize engagement, the framework incorporates evolving user preferences, cognitive states, and fatigue-aware decision-making.

This project was completed as part of the **Recommender Systems** course in the second semester of the M.Tech programme at the **International Institute of Information Technology Bangalore (IIIT-B)**.

## Overview

The framework consists of two primary systems:

- **System 1 (S1):** A fast, two-tower neural recommender that learns initial user and item representations and provides relevance-based recommendations.
- **System 2 (S2):** An adaptive system comprising two complementary components:
  - **S2P – Preference Model:** Uses an LSTM to model evolving user preferences from interaction histories and generate updated user embeddings.
  - **S2F – Cognitive State Model:** Uses Proximal Policy Optimization (PPO) to select recommendations while accounting for user fatigue, discontentment, and relevance.

Together, these systems aim to move beyond static preference prediction towards adaptive, long-term recommendation.

## Architecture and Components

The implementation consists of the following components:

| Component | Description and Purpose |
|---|---|
| **System 1 (S1)** | A two-tower recommender that learns user and item embeddings from IDs and available features. Provides the initial preference representations and supports feature-based cold-start handling. |
| **User Model** | A behavioural simulator that models probabilistic user interactions, preference drift, fatigue, and discontentment. |
| **Synthetic User Dataset** | Generated using the User Model and S1 embeddings. Provides interaction sequences for training the temporal preference model. |
| **System 2 Preference (S2P)** | An LSTM-based model that processes interaction histories to estimate evolving user preferences and update user embeddings. |
| **System 2 Cognitive State (S2F)** | A PPO-based reinforcement learning agent that balances recommendation relevance against fatigue and discontentment to optimize long-term user experience. |
| **Candidate Retrieval** | Uses FAISS (`IndexFlatIP`) for efficient nearest-neighbour retrieval of relevant items and candidate generation. |
| **User Embedding Updater** | Transfers updated preference representations from S2P back into the base user representation, allowing the system to retain learned preferences without repeatedly processing the entire interaction history. |

### How the Components Connect

The overall workflow can be summarized as follows:

1. **Initial representation learning:** S1 is trained on the KuaiRec dataset to obtain user and item embeddings.
2. **User behaviour simulation:** The User Model uses these initial representations to simulate user interactions and generate evolving preference histories.
3. **Preference adaptation:** S2P learns from interaction sequences and updates the user's representation as preferences change.
4. **Cognitive-aware recommendation:** S2F uses the updated preference representation and session-level signals to select items from a candidate pool, accounting for fatigue and discontentment.
5. **Retrieval and embedding updates:** FAISS supports efficient candidate retrieval, while the embedding updater periodically consolidates learned preferences into the base user representation.

The two System 2 components therefore serve different but complementary purposes: S2P estimates **what the user currently prefers**, while S2F determines **what should be recommended considering the user's current state**.

## Dataset

The implementation uses the **KuaiRec dataset** as the foundation for learning initial user and item representations.

A synthetic interaction dataset is additionally generated using the User Model to simulate preference evolution and provide training data for the temporal preference component.

## Technologies and Methods

- **Deep Learning:** Two-tower neural networks and LSTM-based sequence modeling.
- **Reinforcement Learning:** Proximal Policy Optimization (PPO).
- **Similarity Search:** FAISS (`IndexFlatIP`).
- **Dataset:** KuaiRec.
- **Core Concepts:** Sequential recommendation, preference dynamics, cognitive state modeling, fatigue-aware recommendation, and long-term optimization.

## Project Scope

This repository contains the implementation and experimental work carried out for an academic course project. It explores a research-oriented recommender architecture through offline experiments and simulated user interactions.

It is a project prototype rather than a production-ready or independently deployable recommendation service.
