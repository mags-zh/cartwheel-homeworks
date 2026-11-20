# Monitoring `missing_policy_lookup`

## 1. Did the corrected failure estimate move between the two periods?

No, the failure estimate didn't move. Even when the actual scenarios which failed or passed changed (from the monitoring set), the overall value didn't move.

## 2. Do the intervals support a conclusion, or is the result uncertain?

The result is uncertain. The interval is wide, also affected by the judge having a TPR of 0.76. Only 10 conversations were sampled, which is not enough for it to converge on a stable value.

## 3. What did the risk groups reveal that the random estimate did not?

I only had one risk group, the multi-turn scenarios. The judge flagged more of them (6 of 7 in each period, against 3 of 10 random).

## 4. What action should happen if the estimate crosses the threshold?

If the estimate crosses the threshold we need to investigate further by doing error analysis on the new failures, then reviewing and updating the judge. A changed judge must be validated on held-out labels and frozen again before monitoring, and its estimates are not directly comparable with the earlier points.

## Notes on the setup

- Judge `missing_policy_lookup-v1` (`gpt-4o-mini`, frozen in Homework 5). Cartwheel model `gpt-4o-mini`. Random sample 20% (10 of 50 conversations, the same 10 scenarios in both periods), risk group `multi_turn`, threshold 0.25 (chosen before any judge results).
- Langfuse rejects a score without a trace, session or dataset run, so each period's `missing_policy_lookup_corrected_prevalence` score is attached to a small "monitoring run" trace (`monitoring.missing_policy_lookup`, stable id per period) instead of to the batch.
- `.github/workflows/monitor.yml` runs only on a manual trigger (input `hours`, default 24), not on the daily schedule the handout asks for. It runs on a self-hosted runner on the computer that hosts Langfuse, which is started by hand. The Langfuse keys and host come from that runner's own `.env` file, not from GitHub secrets; only the judge's `OPENAI_API_KEY` is a GitHub secret.
- Each score carries its conversation's time (the period score: the end of the window), so the dashboard shows "before" on 2026-10-23 and "after" on 2026-11-19. Langfuse's default dashboard range is one day; choose "Past 90 days" to see both periods.
