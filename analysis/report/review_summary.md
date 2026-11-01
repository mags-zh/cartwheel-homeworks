# Review summary

## Reviewed sample

I reviewed **100 conversations**, made up of 112 Langfuse traces. Cartwheel records one trace per user turn, so the 11 multi-turn conversations span more than one trace. Each conversation counts once: its traces are grouped by `cartwheel.session_id`, and the first turn's trace id is the id used for scores. The population was the 250 final HW3 scenarios (293 traces). No conversation belongs to more than one batch.

| Batch | Selection | Conversations |
| --- | --- | --- |
| 1 | Uniform random (seed 7) | 15 |
| 1 | Cluster representatives over structural features (seed 7) | 15 |
| 2 | Stratified by intent, chosen before looking at outcomes (seed 11) | 30 |
| 3 | Depth searches for the 7 accepted candidate modes, likely positives plus close negatives (seed 17) | 25 |
| 4 | Uniform random from the 165 remaining conversations (seed 23) | 15 |

Composition of the 100:

- Role: 43 shopper, 31 merchant, 26 support.
- Scenario set: 73 coverage, 27 challenge.
- Intent: return or refund 24, product search 15, policy question 15, out of scope 12, order status 11, account change 9, dispute 8, cancellation 6.
- Difficulty: 86 well specified, 12 vague reference, 2 correction across turns.

Open coding produced 121 notes: 87 failure notes and 34 "no failure observed" reviews. Three agent suggestions that I accepted were added later and keep their agent origin.

## Stability check: the final 15

**0 of the final 15 conversations produced a previously unseen consequential mode.** Five were clean. The other ten produced 12 failure notes, and each fits a mode that already existed:

- Seven conversations fit final modes (support-0228, 0108, 0139, 0203, 0066, 0093, 0004).
- Two conversations (support-0045, 0077) fit `wrong_lookup_tool`, first seen in Batch 1. This raised it to three conversations, but I kept it out of the final eight.
- One observation was new: support-0005 guessed a book title without support from the records. I kept it as `unsupported_record_detail` but judged it non-consequential.

Two definitions were broadened to include Batch 4 cases instead of creating new modes: `incorrect_decision_reason` (support-0093) and `unsupported_remedy_suggestion` (support-0203). Neither is in the final taxonomy: I later rejected `incorrect_decision_reason`, and support-0203's note moved to `missing_policy_lookup`.

## Taxonomy revision: `missing_policy_citation`

**Before.** "The reply makes a claim derived from return, refund, cancellation, or store policy without naming the applicable policy identifier." By the end of Batch 4 this mode had 18 notes.

**What I found.** Checking each supporting conversation, I saw that the policy identifier was in the agent's context in only 6 of them. In the others the agent never looked up the policy, so it had no identifier to cite. These need different fixes: retrieving the policy versus citing a policy already retrieved.

**After.** "The reply makes a policy-derived claim without naming the applicable policy identifier even though that identifier is already present in the model-visible conversation or tool results." I moved the other 12 notes (11 conversations) to `missing_policy_lookup`. New boundary: if the identifier is missing because the policy was never retrieved, the failure is `missing_policy_lookup`; if the identifier was available and the reply left it out, it is `missing_policy_citation`.

**Search again.** Batch 3 had already searched for the old definition. After the revision, the agent scanned all 100 reviewed conversations for tool results containing a policy id, followed by a policy-based reply that omitted it. It returned 6 suggestions:

- Accepted: support-0064, support-0180 (the 45-day policy claim) and support-0209.
- Rejected: support-0180's tool-status reply, which reports a tool result rather than a policy claim.
- Rejected: support-0169, where the policy ids in context were not the policy that applies.
- Rejected: support-0175, where the reply should not have answered at all because the request was out of scope. That is an `out_of_scope_not_refused` failure, not a missing citation.

The mode now has 9 supporting conversations, and 9 Present labels in Part E.

## SPEC.md revisions

- **RESP-6** (when the request is explicit, the target is unambiguous and all inputs are available, act without re-asking). Motivating annotations:
  - support-0064, `annotation-1793353091663`
  - support-0035, `annotation-1793353426766`
  - support-0031, `annotation-1793354149278`
  - support-0188, `annotation-1793354246807`
- **ESC-5** (when authorized records contain a material contradiction or an impossible value, stop the affected action, don't present the value as trustworthy, explain that the records need review, and escalate). Motivating annotations:
  - support-0227, `annotation-1793357133614`: delivered before it was shipped.
  - support-0233, `annotation-1793368746004`: product belongs to a different store.
  - support-0246, `annotation-1793374606998`: negative price.
  - support-0228, `annotation-1793392559656`

## Final modes and sample fractions

Every final mode was judged Present or Absent on every reviewed conversation: 8 × 100 = 800 judgments, each written to Langfuse as a score and to `analysis/state/labels/<mode>.jsonl`. These are **sample fractions, not prevalence estimates**. Batches 2 and 3 deliberately changed the sample's composition, and Batch 3 targeted likely failures.

| Mode | Requirement | Present | Sample fraction |
| --- | --- | --- | --- |
| `missing_policy_lookup` | PURPOSE-1, SCOPE-1 | 38 | 38% |
| `unnecessary_process_narration` | RESP-5 | 29 | 29% |
| `missing_policy_citation` | RESP-1 | 9 | 9% |
| `missing_required_account_escalation` | ESC-2 | 9 | 9% |
| `asks_instead_of_acting` | RESP-6 | 8 | 8% |
| `store_identifier_mismatch` | TOOL-3 | 7 | 7% |
| `data_inconsistency_not_escalated` | ESC-5, RESP-3 | 7 | 7% |
| `out_of_scope_not_refused` | SCOPE-2 | 3 | 3% |
