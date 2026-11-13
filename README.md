# Superagent vs Multiagent Benchmark

**Status:** reproducible *illustrative* Monte Carlo experiment, not a validated Anthropic billing emulator.

## Question
For 1,000 clients and 200 replies over eight hours, is one shared, growing agent conversation cheaper than independent client sessions?

## Reproduce
```bash
pip install numpy
python simulate.py --seed 20261113 --runs 1000 --output results.csv
```

## Assumptions
- 1,000 clients, 200 replies in 480 minutes; event times uniformly distributed.
- Client popularity sampled from a Zipf-like distribution, with exponent varying per run.
- Common prefix: 30,000 tokens; client context: 1,200 tokens.
- Per reply: 200 new input, 300 visible output, 400 thinking tokens billed as output.
- 500 tokens added to history (user + visible assistant), **thinking not retained**.
- No compaction. Common header cached for 5 minutes in both architectures.
- Separate client history: 5-minute TTL; shared growing history: 1-hour TTL.
- Illustrative USD per million tokens: input 4, output 20, cache read 0.20, cache write 5m 5 / 1h 8.

## Results: 1,000 runs, seed 20261113
| Metric | Independent sessions | Shared superagent |
|---|---:|---:|
| Median cost | $9.20 | $10.85 |
| 5th percentile | $8.57 | $10.13 |
| 95th percentile | $9.91 | $11.57 |

Shared superagent cheaper in **0 of 1,000 runs** under these assumptions. Median shared cost is approximately 18% higher. These percentiles describe variation in synthetic workloads, **not uncertainty about production costs**.

## Known limitations
1. Cache behavior is simplified and does **not** implement Anthropic's exact breakpoint rules, token thresholds, or automatic caching semantics.
2. The shared architecture bills the 1,200-token per-client context as uncached input every turn. Different prompt ordering can change this.
3. A separate client's entire tail is treated as a cache write after 5-minute inactivity, rather than modeling precise prefix reuse.
4. Exact extended-thinking context retention depends on API configuration.
5. Uniform event times, synthetic client skew, and fixed token lengths may not resemble production traffic.
6. Average response latency is modeled separately in `latency_model.py` using an illustrative M/D/1 queue approximation, not measured API timings. Privacy isolation, context pollution, answer quality, and rate limits are not empirically evaluated.
7. The price assumptions should be checked against current model-specific Anthropic pricing before making any financial decision.

## Next experiment
Replay identical 200-turn workloads using actual API requests with explicit cache breakpoints; record `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`, latency, and answer quality. Include gaps longer than 5 and 60 minutes.

## Sources
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [Preble, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html)


## Exact simulation algorithm

Each run uses a seeded NumPy random generator. It samples 200 event timestamps uniformly over 480 minutes and sorts them. It samples a client-popularity exponent uniformly between 0.5 and 1.4; each of the 200 events selects a client with probability proportional to `(client_index + 20)^(-exponent)`. Therefore the experiment has many inactive clients, some repeat customers, and uneven per-client history lengths. This is **not** a fitted real-world activity distribution.

For each event, the simulator tracks five billable buckets separately for each architecture: regular input, cache reads, 5-minute cache writes, 1-hour cache writes, and output. A shared 30k-token prefix is read when the previous event was at most five minutes ago, otherwise written at the 5-minute write rate. The independent architecture tracks 1,200 tokens plus the accumulated 500-token history of each client and reads or rewrites that tail according to the time since **that client's** previous event. The shared architecture tracks the total accumulated history across all events, reads it if the previous global event was within 60 minutes, and otherwise rewrites it at the one-hour rate. In addition, it charges the selected client's 1,200-token profile as uncached input on each shared-agent request.

The model bills 200 regular input tokens and 700 output tokens (300 visible + 400 thinking) per event for both architectures. It grows retained history by 500 tokens (200 + 300), **not** by 900; thinking is billed but assumed not retained. The synthetic cost formula is:

`cost_USD = (4 * input + 0.20 * cache_read + 5 * write_5m + 8 * write_1h + 20 * output) / 1_000_000`

The common-prefix clock is updated on every event. The shared-history clock is updated on every event. The per-client clock is updated only on that client's events. These clocks are a simplification, **not** a guarantee of actual cache TTL renewal behavior.

## Interpretation and sensitivity

The experiment fixes **200 replies total**, not 200 replies per client. A single shared history therefore reaches 100,000 retained tokens by the end, while most individual client histories remain short. It assumes a shared agent can access the relevant client's profile without contamination from other clients, but it does **not** test that assumption.

The result depends on cache TTL, prompt ordering, whether a cacheable prefix actually matches, the distribution of interarrival times, retained context size, output token counts, and pricing. The toy model should not be used to claim the optimal production architecture. A true Anthropic comparison must inspect real request-level usage counters and cache breakpoint semantics.

## Reproducibility checks

Use Python 3.10+ and NumPy. The published script's default seed is `20261113`. The CSV has 1,000 data rows with columns `run,separate_usd,shared_usd`, indexed from 0 to 999. Summary values are rounded to two decimals in this README. The code, not the rounded table, is the authoritative specification of this synthetic experiment.


## Metrics and scope

The benchmark tracks **cost**, **average response latency**, and **reasoning / task success quality**. Average response latency is the only timing metric; there is no separate “execution speed” metric. `latency_model.py` provides a simplified queueing estimate using a 20-second assumed service time and 200 requests per eight hours, **not measured API latency**. Reasoning and task success remain proposed evaluation dimensions, not measured outcomes. The benchmark does not establish an overall winner.
