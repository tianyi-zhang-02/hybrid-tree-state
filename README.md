# Caching hybrid-model state in tree rollouts

2026-11-24. All timings were taken on one RTX 4090 with Qwen3.5-9B (online FP8) on SGLang 0.5.23. A cache-fidelity check was also run on vLLM 0.34.0, and a state-rebuild test in HF Transformers.

## 1. Summary

This note is for people who run tree-shaped generation on hybrid models: tree rollouts in RL, beam or tree search, MCTS-style search over reasoning steps, and agent search with tool calls. Hybrid models keep a recurrent state in some layers; examples are Qwen3.5, Qwen3-Next and Nemotron-H.

We asked whether the tree program's own knowledge should decide which states the engine keeps: which nodes it will expand next, and which it has pruned. On four workloads that gains at most about 2% over one simple rule, holding the root.

- **One tree at a time.** SGLang's default LRU is at most 4.4% slower than the best policy at the same budget (7.1% in the worst seed).
- **Several trees at once, as in a rollout batch.** LRU is 7–18% slower than the best policy, because it evicts tree roots and children then recompute the whole prompt. Holding each tree's root with a session for the life of the tree removes every such miss. That leaves 2–8% to the no-loss floor.
- **Search that returns to old nodes (MCTS-like).** LRU is 0.5–3.8% slower than the best policy. Holding the root alone comes within 0.5% of an oracle that knows which nodes the search will revisit.
- **Agent search with tool outputs of up to 3k tokens per step (SWE-Search-like).** Here a miss below the root costs thousands of tokens, which is the case most favourable to smarter retention. LRU is still only 1.0–2.5% slower than the best (4.3% in the worst seed).
- **Program hints add little.**
  - Over holding the root alone, they gained at most 2.3% in the forests, 0.7% in search and 1.6% in agent search.
  - At tight budgets, holding every expandable node was worse than the default.
- **Prefix caching of hybrid state is exact** on both engines in the plain setting: a cached state continues exactly as a recompute would. The numbers change only with the computation path, by the same amount as the existing gap between generation and a trainer's recompute.
- **Approximate state rebuild is not exact.** Replaying the last 5–10% of the prefix, as Tail-Replay does, moves per-token log-probs by 0.11–0.22 nats, 8–16× the gap with the exact state.

What to do:

- **Rollout frameworks:** hold each prompt's state with a session until its tree or group is done (section 6). It needs no engine change.
- **Do not turn prefix caching off for hybrid models on fidelity grounds alone.** The exception is speculative decoding (MTP), which we did not test.
- **Do not use tail replay in place of an exact state in RL rollouts** without measuring the log-prob gap it adds (section 8).

## 2. Why hybrid models need their own answer

- An attention model can reuse a cached prefix cut at any token.
- A hybrid model also needs the recurrent state at the cut point. That state exists only where the engine kept a checkpoint.
  - SGLang 0.5.23 with `extra_buffer` (the default when the overlap scheduler is on) keeps one at prefill chunk boundaries, near the end of each prefill (within 64 tokens in our runs), and every 256 tokens during decode. It does not keep the exact end of a decode, so a child that forks there recomputes up to 255 tokens. SGLang main keeps exact ends ([sgl-project/sglang#49886](https://github.com/sgl-project/sglang/pull/49886)).
  - vLLM 0.34.0 with `align` keeps one at block boundaries, which are 576 tokens for this model.
- Checkpoints live in a fixed pool and are evicted by LRU.
- A request whose path has lost its checkpoints recomputes from the last one it finds. For a branch under a long prompt, that can mean the whole prompt.
- Tree rollouts make this frequent: every branch starts from a parent's state, and many branches share one long prompt.

## 3. Setup

**Engine.**

- SGLang 0.5.23 with `extra_buffer` and `--enable-session-radix-cache`, FP8 online quantization, page size 1.
- Running limit of 8 requests. Each running request takes 4 slots of the state pool.
- The budget is the number of slots beyond those: 8, 16 or 31. The engine keeps 2 slots empty, so 6, 14 or 29 states stay cached while 8 requests run.

**Workloads.** Each root is random token ids: 8,000 or 32,064 tokens, or 4,160 for agent search. Each workload ran with 3 seeds, and the policy order was rotated by seed.

| Name | Shape | Modelled on |
| --- | --- | --- |
| Segment tree | Each node decodes 256 tokens and branches 2 ways. Depth 6, pruned to 8 nodes per level. No revisits | TreePO; TreeRL and ARPO use branch factor 2 |
| Forest | 4 segment trees with 8k roots, or 2 with 32k roots, started together and sharing the pool | Several prompts in one rollout batch |
| Search with revisits | Nodes are reasoning steps of 128–256 tokens with fixed scores. Each of 16 simulations picks a node by WU-PUCT, then rolls out 2 children per step down to a terminal node, 4 simulations at a time. A tree has 104–128 requests and 15 revisits, to nodes created 6–104 nodes earlier | Adaptive Parallel MCTS ([arXiv 2604.00510](https://arxiv.org/abs/2604.00510)) |
| Agent search | A node is an agent step: an action of 100–400 decoded tokens, then a tool output of 100–3,000 tokens (median about 1.4k) that later steps read. 100 iterations each pick one node by UCT over the whole tree and add one child, 4 at a time. Trees reach depth 13–14 and hold 139–178k tokens. 28–40 expansions per tree go back to a node made 8 or more nodes earlier | SWE-Search ([arXiv 2410.20285](https://arxiv.org/abs/2410.20285)), LATS ([arXiv 2310.04406](https://arxiv.org/abs/2310.04406)) |

**Policies.**

| Policy | What it holds |
| --- | --- |
| Default | Nothing; the engine's LRU decides |
| Root | The root's state, held with a session until the tree is done |
| All expandable | Every node that may still be expanded |
| Search-ranked | The root, plus the candidates the next round of selection would pick first, up to what fits |
| Oracle | The root, plus the nodes the search will actually revisit, soonest first. A reference only, since it uses the future |

**Measures.**

- Completion time of the tree or forest.
- Recomputed tokens: each request's prompt length minus its cached tokens.
- The floor: for each request, the fewest tokens any run recomputed for it, summed over the tree.

## 4. Results

Each cell is completion time relative to the best policy in the row, as a mean over 3 seeds. "Program hints" is "All expandable" for the segment trees and forests, and "Search-ranked" for both kinds of search.

| Workload | Budgets | Default | Root | Program hints |
| --- | ---: | ---: | ---: | ---: |
| Segment tree, 8k | 8 / 16 / 31 | best / +0.1% / best | not run | +0.2% / best / +0.2% |
| Segment tree, 32k | 8 / 16 / 31 | best / +4.4% / best | not run | +1.8% / best / +0.2% |
| Forest, 4 × 8k | 8 / 16 / 31 | +13.6 / +12.0 / +7.2% | best / best / best | +18.7 / +8.3 / +5.2% |
| Forest, 2 × 32k | 8 / 16 / 31 | +18.2 / +13.2 / +11.5% | best / best / +2.3% | +22.0 / +5.1% / best |
| Search, 8k | 8 / 16 / 31 | +0.5 / +1.3 / +1.0% | best / +0.5% / best | — / +0.5 / +0.4% |
| Search, 32k | 8 / 16 / 31 | +3.8 / +2.4 / +2.3% | best / best / +0.7% | — / +0.0% / best |
| Agent search, 4k | 8 / 16 / 31 | +1.0 / +1.3 / +2.5% | +0.1 / +0.9 / +1.9% | best / +0.4 / +0.3% |

Notes on the table:

- **Segment tree.** The default's one real loss, 4.4% at 32k with 16 states (7.1% in the worst seed), comes from a single miss per tree that recomputes the whole 32k root.
- **Forest.** The remaining gap from "Root" to the floor is 6–8% with 4 × 8k and 2–7% with 2 × 32k.
- **Search.**
  - The oracle is within 0.5% of the best in every row.
  - At budget 8 there is no room for candidates beyond the root, so the program-hint policies equal "Root" there.
  - With the root held, every policy is within 1% of the floor's time.
- **Agent search.**
  - The oracle is best at budgets 16 and 31, and at 31 it reaches the floor.
  - The best policy is within 3.5% of the floor at every budget.
  - Under the default, 3–6 requests per budget fell back into the root, out of about 570 over 3 trees. None did with the root held.

## 5. Why holding the root is enough here

**Time is linear in recomputed tokens.**

- In the forests a straight line fits all 27 runs per root length with R² 0.99. The slope is 16.3k recomputed tokens per second at 8k and 13.8k at 32k.
- In search, a fit within each seed gives 19.6k tokens per second at 8k (R² 0.61; the policies differ so little there that the fit is loose) and 14.3k at 32k (R² 0.96).
- In agent search, the same fit gives 16.2k tokens per second (R² 0.94).
- So a policy matters only through the tokens it makes the engine recompute. Ordering and queueing add nothing measurable.

**A miss costs the length of the path it has to recompute.**

- In the first three workloads the only long segment is the root. Below it, a step is 128–256 tokens, and a whole path is under 2k tokens.
- Under pressure, LRU evicts the oldest state first, and the root is always the oldest. So the miss LRU cannot avoid is also the most expensive one: the whole prompt.
- Holding the root turns that miss into a recompute of a few short segments.

**Below the root, LRU already does well.**

- In search, the nodes near the root are shared by many simulations.
- Each reuse refreshes a node in the LRU, so the upper tree stays cached even when the search returns to old nodes.
- The misses that remain recompute a few hundred tokens, a fraction of a second against 74–83 s per tree.

**When this would change, and what agent search showed.**

- Program hints can pay only when two things hold:
  - the paths below the root are long, so that a miss there is expensive;
  - they are revisited after a long gap.
- Agent search with long tool outputs was built to meet both conditions. We expected LRU to fall 5–20% behind, but it fell only 1.0–2.5% behind. Three things kept the loss small:
  - **Few misses.** UCT mostly expands recent nodes. Under LRU only 9–14% of expansions found their parent's state gone.
  - **Short fallbacks.** A miss usually falls back two or three steps, recomputing a median of 4–5k tokens. Ancestors that many descendants share stay cached, so a miss rarely reaches the root.
  - **Decoding dominates.** It sets the time: about 116 s per tree, against 3–5 s of extra recompute under LRU.
- Holding what the search will expand next removes most of that small loss (1.6% at the largest budget), but not more.

## 6. Holding the root on SGLang

SGLang's session radix cache ([sgl-project/sglang#29173](https://github.com/sgl-project/sglang/pull/29173), in 0.5.23) lets a request carry a `session_id`. When the request finishes, the cache entry it ended on, including the Mamba state, is protected until the session closes. A rollout needs the server flag and two calls around each prompt's tree or group:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --enable-session-radix-cache ...
```

```python
import requests

base = "http://localhost:30000"

# Prefill the prompt once with a session id. The engine keeps its state protected.
requests.post(f"{base}/generate", json={
    "input_ids": prompt_ids,
    "sampling_params": {"max_new_tokens": 1},
    "session_id": "prompt-17",
})

# ... send every branch as usual, WITHOUT the session id. They match the prompt
# and reuse its held state.

# When the tree or group is done:
requests.post(f"{base}/close_session", json={"session_id": "prompt-17"})
```

- **Send only the prompt with the session id.** Within one session, protection moves to the newest entry the session's requests end on. A branch sent with the same id would take the protection off the prompt when it finishes.
- **Protection is soft.** Held states are evicted only after all unheld ones, so holding more than the pool can keep defeats the rule. In the forests, holding every pending node made things worse at tight budgets: LRU among held states then dropped the roots first.
- **Within the pool it holds.** In agent search, held parents were selected 1,262 times over all runs, and one of them found its state gone.
- Hold one state per tree, the root, and add more only if you know they fit.

## 7. Cache fidelity

Some RL frameworks turn prefix caching off for hybrid models because cached state changed the training numbers. Examples are NeMo-RL ([NVIDIA-NeMo/RL#4622](https://github.com/NVIDIA-NeMo/RL/pull/4622), which sets `enable_prefix_caching: false` for this reason) and SkyRL ([skyrl-project/SkyRL#1981](https://github.com/skyrl-project/SkyRL/issues/1981)), both on vLLM. We compared cached and recomputed continuations on 16 natural-text prompts of 8k and 32k tokens.

**SGLang 0.5.23.**

- Continuing from a cached state is exactly the same as a recompute that splits the prefill at the same position.
- What changes the numbers is the computation path:
  - where prefill is split: splits at multiples of 2,048 agree with each other, others do not;
  - whether tokens went through decode or prefill.
- The change averages about 0.05 nats per token. That is the same as the gap between generation and recompute with no cache at all. Generations that started warm and cold cannot be told apart in that gap.

**vLLM 0.34.0 (`align` mode).** Warm generations match cold ones exactly. It reuses only block-aligned prefill states, which are coarser but exact.

**Not tested:**

- concurrent batches;
- bf16;
- speculative decoding (MTP);
- Mamba2 models.

SkyRL's report involves MTP, and its issue points at the state boundary after accepted draft tokens. That is a separate question from caching itself.

## 8. Approximate state rebuild

Tail-Replay and SuffixReplay rebuild a lost recurrent state by replaying the last part of the prefix from a zero state, with the attention KV kept exact. That would make misses cheap and shrink any gain from retention. Their papers report task scores only, so we measured the per-token log-probs that RL uses.

**How.**

- Qwen3.5-9B in bf16 on HF Transformers, following Tail-Replay's description, on the same 16 roots.
- 256 tokens are sampled at temperature 1. The trainer's view is the same tokens teacher-forced from the exact state.
- Both directions were run: tokens sampled from the exact state, and tokens sampled from the rebuilt one.
- Replaying 100% of the prefix gives bit-identical log-probs, so the harness itself adds nothing.

**Results.**

- At the papers' 5–10% replay, the mean |log r| is 0.11–0.22 nats per token, against 0.014 with the exact state: 8–16× larger.
  - Here r is the trainer's probability of a sampled token divided by the generator's.
  - 17–31% of tokens fall outside PPO's clip range [0.8, 1.2], against 0.1%.
- 20% replay still gives 0.10–0.13 nats.
- The error lasts through the 256 tokens. At 5% replay, the last eighth of them is still 10× the baseline or more.

**What it means.** For RL rollouts, tail replay at the published fractions is not a drop-in for an exact state. A framework that used it would rely on its off-policy correction, which would then clip a fifth or more of the tokens. We did not test whether training notices.

## 9. Limits

- One GPU, one 9B model in FP8, and a running limit of 8.
- Synthetic tree shapes with random-token roots and tool outputs, and 3 seeds per cell.
- Times are tree completion times, not RL step times.
- Untested: larger batches of prompts, larger running limits, routing across GPUs, and SGLang main's exact checkpoints.
- The rebuild test is an emulation, not the papers' code or SGLang's kernels, and measures the mismatch only.

## 10. Reproducing

Scripts are in `scripts/`:

- `tree_shapes.py`: seeded tree shapes (T1 segment trees, T2 search, T3 agent search);
- `serve_sglang.sh`: the server settings; `probe_capacity.py`: measures how many states the pool keeps;
- `replay_tree.py`, `replay_forest.py`, `replay_t2.py`, `replay_t3.py`: replay a shape against a server under a policy;
- `run_trees.sh`, `run_forest.sh`, `run_t2.sh`, `run_t3.sh`: the sweeps;
- `summarize_forest.py`, `summarize_t2.py`, `summarize_t3.py`: the tables;
- `mock_server.py`: a fake server that imitates the session and eviction behaviour, to check a policy's control flow without a GPU;
- `cache_fidelity.py` and `cache_fidelity_vllm.py`: the fidelity test;
- `tail_replay_drift.py` and `summarize_tail_replay.py`: the rebuild test.

The sweep scripts were written for one WSL machine. Before timing anything they wait until GPU use is low, no other model job is running, and no Windows process matching `GATE_RE` (empty by default) is open.

Raw results are in `runs/`.
