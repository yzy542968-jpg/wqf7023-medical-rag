# V10 Evidence Answer Normalization Amendment

Status: frozen after inspecting early execution format behavior, but before
full Validation completion, aggregate policy metrics, and policy selection.

Revision 1 produced structurally complete rows, but early outputs showed that
MedGemma could ignore the requested two-sentence limit and continue listing
findings until the 64-token ceiling. This amendment enforces the already stated
answer-length contract deterministically.

The authoritative answer is normalized before metric calculation as follows:

1. discard text after `<end_of_turn>` or `<unused94>thought`;
2. collapse whitespace;
3. retain the first two complete sentence units ending in `.`, `?`, or `!`;
4. if no complete sentence exists, retain the normalized available text;
5. if no answer text exists, use the frozen explicit inability statement;
6. recompute deterministic historical provenance using the normalized answer;
7. recompute all answer metrics from the normalized answer.

Raw generations, token-ceiling indicators, and Revision 1 rows are retained
unchanged. The finalizer refuses to run until all 2,256 planned Revision 1 rows
exist. Only finalized rows may select E0/E1/E2 or enter the confirmation
configuration. This amendment does not alter model weights, prompts, evidence,
retrieval, cases, references, or Test data.

