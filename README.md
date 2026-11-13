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
6. Latency, queueing, privacy isolation, context pollution, answer quality, and rate limits are excluded.
7. The price assumptions should be checked against current model-specific Anthropic pricing before making any financial decision.

## Next experiment
Replay identical 200-turn workloads using actual API requests with explicit cache breakpoints; record `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `output_tokens`, latency, and answer quality. Include gaps longer than 5 and 60 minutes.

## Sources
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [Preble, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5bc342f48de8264779952fac378f96dc-Abstract-Conference.html)
