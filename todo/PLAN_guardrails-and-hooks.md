# PLAN: Guardrails and Hooks

## Status

Done

## Goal

Add additive, provider-neutral guardrail APIs to KKit. Registered message guardrails inspect every message role with conversation/run context, may allow or block, and can stop live output mid-stream. Also harden tool authorization and bounded-run behavior. Keep telemetry hooks observational and keep application-specific policy engines, persistence, moderation vendors, and UIs outside KKit.

## Success Criteria

- Applications register ordered message guardrails once through `CompletionCallInput`; agents inherit them through embedded config.
- Guardrails inspect system, user, assistant, and tool messages plus assistant content/reasoning streams and tool-call arguments.
- Each guardrail receives an immutable candidate message, cloned conversation context, phase/channel, model, round, and provider attempt.
- Only explicit `allow` or `block` decisions exist. Any block or guardrail error fails closed, aborts the completion, stops remaining tool calls, and returns a typed error.
- Live deltas remain live. Candidate accumulated content is checked before each delta reaches observer hooks; the threshold-crossing delta is not emitted.
- A blocked partial assistant response is returned to the caller but not appended to agent history.
- KKit ships a configurable system-prompt leakage matcher as an attachable guardrail plus a runnable example.
- Central tool policy, validated approval, and aggregate limits prevent bypass-prone or runaway framework execution.
- Existing documented v1 APIs remain available; baseline and new tests pass under race and vet checks.

## Confirmed Decisions

- Initial scope combines execution hardening with generic message guardrails.
- API evolution is additive in v1; no broad hook replacement or v2 break.
- Message guardrails run for every message role, not assistant output only.
- Decisions are `allow` or `block`; guardrails do not mutate, redact, or replace content.
- Registered guardrails compose with strictest-wins semantics: evaluate in registration order and stop at first block/error.
- System-prompt leakage detection is a bundled attachable guardrail, not hardcoded completion behavior.
- Streaming remains live and stops midway when the accumulated candidate crosses policy.
- Blocked partial output is returned but excluded from persistent agent history.
- Application tool policy and tool metadata compose strictest-wins. Tool-policy denial aborts the completion and prevents remaining calls in the assistant response.

## Current State

- `pkg/llm/hooks.go` mixes observer callbacks with abortable `OnEvent`; focused callbacks run before matching `OnEvent` checks.
- OpenAI streaming forwards content/reasoning deltas immediately through `CompletionHooks`. Final assistant checking occurs later in `Completion`.
- Tool approval is declared by each `Tool`, evaluated in `Toolbox.CallWithInfo`, and occurs before typed argument decoding. Unknown policy enum values silently behave like no approval.
- Exported `Tool.Call` bypasses toolbox approval when invoked directly. Framework guarantees currently cover completion/toolbox paths only.
- Approval denial becomes retryable model feedback unless `ToolErrorInterceptor` aborts. Existing limits cover tool rounds, provider retries, error length, and caller context only.
- `v1.8.1` baseline passes `GOMODCACHE=/tmp/kisaragi-kit-go-mod-cache GOCACHE=/tmp/kisaragi-kit-go-build-cache go test ./...`.

## Implementation Tracker

- [x] Add message guardrail contracts, validation, ordered evaluation, and typed errors in `pkg/llm`.
- [x] Guard initial messages, live assistant content/reasoning, final assistant messages/tool arguments, tool results, and approval decision messages.
- [x] Preserve safe streamed prefixes in returned output while excluding blocked candidates from agent history.
- [x] Add bundled system-prompt leakage guardrail under `pkg/llm/guardrail`.
- [x] Add central tool policy, validated approval flow, and execution budgets.
- [x] Add focused core, agent, toolbox, provider, and matcher tests.
- [x] Add `examples/guardrails` and update README/package docs/changelog.
- [x] Run formatting, unit, race, vet, and static analysis verification; record results.

## Implementation Plan

### 1. Public message guardrail contract

- Add to `pkg/llm`:
  - `MessageGuardrail` interface with `Name() string` and `CheckMessage(context.Context, MessageGuardrailInput) (MessageGuardrailDecision, error)`.
  - `MessageGuardrailFunc` adapter so applications can register ordinary functions with a stable name.
  - `MessageGuardrailPhase`: `input`, `assistant_content_delta`, `assistant_reasoning_delta`, `assistant_reasoning_final`, `assistant_final`, `tool_result`, and `approval_decision`.
  - `MessageGuardrailInput` containing `Message`, cloned `Messages`, `Phase`, `Model`, `Round`, and `Attempt`. The candidate is separate from prior context and never aliases mutable completion state.
  - `MessageGuardrailAction` values `allow` and `block`; zero/unknown actions fail validation and therefore fail closed.
  - `MessageGuardrailDecision` containing `Action` and non-sensitive `Reason`.
  - `ErrMessageGuardrailBlocked` and `MessageGuardrailError` carrying guardrail name/index, phase, action/reason, and wrapped cause without copying candidate/system content into `Error()`.
- Add `MessageGuardrails []MessageGuardrail` to `CompletionCallInput`. Validate nil entries and blank/duplicate names before provider work begins.
- Evaluate in registration order. Continue only on explicit allow; first block or callback error stops evaluation and completion.

### 2. Message checkpoints and streaming flow

- Before first provider attempt, check every message in `CompletionCallInput.Messages` in order with phase `input`. This includes system, user, assistant, and tool roles. Generated messages are checked at creation and are not redundantly rechecked on every later provider round.
- For each provider attempt, wrap completion hooks with attempt-local accumulators before passing them to `ChatClient`:
  1. append proposed content/reasoning delta to a candidate buffer;
  2. run guardrails against an assistant `Message` containing the full proposed accumulated channel;
  3. on allow, commit buffer and forward the original observer delta/event;
  4. on block/error, do not forward the threshold-crossing delta and cancel/close the provider stream.
- Keep content and reasoning accumulators separate. Phase identifies which assistant channel is being checked.
- After a provider returns, check one `assistant_final` candidate containing final content and tool calls. The final check covers non-streaming/custom clients and tool-call arguments even when no deltas were emitted.
- Before appending each tool result to provider/output history, check the tool-role message with phase `tool_result`.
- Guardrails run before legacy observer callbacks for protected message data. Existing `CompletionHooks.OnEvent` remains available but is documented as lifecycle abort/telemetry, not the primary message-policy surface.

### 3. Abort and transcript semantics

- Guardrail block/error is non-retryable: provider retry logic recognizes `ErrMessageGuardrailBlocked` and returns immediately.
- Track the last fully allowed streamed content prefix outside the provider adapter. On mid-stream block, return non-nil `CompletionCallOutput` with that prefix in `Content` and any already completed prior-round messages/tool calls.
- Never append the blocked assistant candidate to `CompletionCallOutput.Messages`. `Agent.complete` therefore persists earlier completed round messages but not the blocked partial candidate; returned `Content` remains caller-visible for already emitted UI reconciliation.
- Initial-message block returns no provider call and no generated output. Tool-result block aborts before the result reaches model history and stops all remaining tool calls.
- Preserve `errors.Is(err, ErrMessageGuardrailBlocked)` and `errors.As(..., *MessageGuardrailError)` across provider, completion, and agent wrapping.

### 4. Bundled system-prompt leakage matcher

- Create `pkg/llm/guardrail` with `NewSystemPromptLeakGuardrail(SystemPromptLeakConfig) (llm.MessageGuardrail, error)`.
- Config fields:
  - `Threshold float64`: fraction `(0,1]` of any one system message covered by exact normalized word sequences; zero uses `DefaultSystemPromptLeakThreshold = 0.20`.
  - `MinMatchWords int`: minimum contiguous word sequence counted as evidence; zero uses `DefaultSystemPromptLeakMinMatchWords = 8`; negative is invalid.
- Normalize with standard-library Unicode letter/digit tokenization, lowercase, and whitespace/punctuation removal. Do not use embeddings, provider calls, stemming, or fuzzy semantic claims.
- Build word shingles of `MinMatchWords`; mark system-prompt token positions covered by shingles found in the candidate; block when unique covered positions divided by system-message token count reaches threshold. For shorter prompts, require the complete normalized prompt.
- Compare each system message independently. Inspect assistant content, accumulated reasoning, and serialized tool-call arguments; ignore non-assistant candidates so input system/user/tool messages do not self-trigger.
- Return only matched ratio, configured threshold, and guardrail name in the decision/error metadata; never echo matched prompt text.
- Add `examples/guardrails/main.go` showing registration, a 20%/8-word configuration, live delta output, typed block handling, and safe partial-output reconciliation.

### 5. Tool authorization and bounded execution

- Add application-owned `ToolPolicyHook` to `Toolbox`; each request includes tool-call ID when available, name, validated arguments, declared policy, round, and model context.
- Add decisions `allow`, `require_approval`, and `deny`. Merge tool-declared approval with application policy using `deny > require_approval > allow`; callback errors fail closed.
- Validate arguments against the existing tool JSON schema and canonicalize them before central policy or human approval; preserve `Tool` struct layout for v1 source compatibility. Typed handler decoding and execution remain after authorization. Approval receives validated canonical JSON.
- Validate approval mode, risk, preview, and combinations at registration. Unknown values fail registration. Keep legacy `Call`/`CallWithInfo` delegating through the protected path; document exported direct `Tool.Call` as outside guarantees and deprecate it without removal in v1.
- Policy denial returns typed non-retryable error, aborts the completion, and skips remaining tool calls.
- Add aggregate optional limits to `CompletionCallInput`: total provider attempts, total tool calls, repeated identical tool calls, approval denials, and reported total tokens. Zero uses documented defaults; negative values fail validation. Canonical JSON identifies repeated calls. Missing usage cannot satisfy a configured hard token ceiling, so enabling that ceiling with an unmetered response returns a typed unsupported-budget error before another round.

### 6. Compatibility and documentation

- Preserve `CompletionHooks`, `OnEvent`, `ApprovalHook`, `ToolErrorInterceptor`, existing constructors, and legacy toolbox call methods. New controls are additive fields/options and protected call paths.
- Do not change provider-neutral `Message` storage shape, wire protocols, persistence, or UI APIs.
- Document trusted boundary: guardrails are in-process privileged code and receive system messages; KKit protects only data and calls routed through its completion/toolbox APIs.
- Document streaming limitation: content allowed below threshold was already emitted and cannot be recalled; matcher stops the crossing delta, not prior accepted deltas.

## Edge Cases and Failure Behavior

- Empty candidate content still reaches guardrails when tool calls exist; prompt leakage inside tool arguments remains detectable.
- Duplicate guardrail names, nil guardrails, invalid decisions/config, callback errors, and context cancellation fail before further side effects.
- A final guardrail may block content previously allowed delta-by-delta if its whole-message policy differs; returned safe prefix remains caller-visible but is not persisted.
- Multiple system messages use per-message coverage, preventing one long prompt from diluting leakage from a short prompt.
- Common short phrases below `MinMatchWords` do not count. A system prompt shorter than the minimum requires full normalized equality/subsequence coverage.
- Reasoning is checked and streamed independently from assistant content, but remains absent from persisted `Message` because current KKit transcripts are text-content/tool-call only.
- Custom `ChatClient` implementations that emit no deltas still receive final-message protection; they cannot provide mid-stream stopping without using KKit hooks.
- Guardrail panics remain caller panics; KKit does not silently recover from application code.

## Tests and Verification

- `pkg/llm`: registration validation, every message role/phase, immutable context copies, order/strictest-wins, callback errors, typed error matching, retry suppression, multi-tool stop, tool-result block, final-only custom client, safe-prefix output, and transcript exclusion.
- `pkg/llm/provider/openai`: guardrail runs before content/reasoning observer callbacks and threshold-crossing deltas are not forwarded.
- `pkg/llm/agent`: blocked partial content returned but not persisted; earlier completed-round messages remain persisted exactly once.
- `pkg/llm/guardrail`: normalization, punctuation/case, exact shingles, disjoint coverage, threshold boundary, short prompts, multiple system messages, reasoning/content/tool arguments, false-positive minimum, invalid/default config, and no sensitive error text.
- `pkg/llm/tool`: central-policy precedence, schema-validated arguments before approval, canonical approval payload, fail-closed policy errors, legacy protected methods, invalid approval enum rejection, and budget counters.
- Run `gofmt` on changed Go files.
- Run `GOMODCACHE=/tmp/kisaragi-kit-go-mod-cache GOCACHE=/tmp/kisaragi-kit-go-build-cache go test ./...`.
- Run same cache environment with `go test -race ./...` and `go vet ./...`.

## Completion Notes

- Implemented additive message guardrails across initial messages, streaming content/reasoning, final assistant responses/tool arguments, tool results, and approval decision messages.
- Added safe-prefix mid-stream blocking for emit-helper and direct-callback `ChatClient` implementations; OpenAI streams now close on early return.
- Added `pkg/llm/guardrail` system-prompt word-shingle matcher and `examples/guardrails`.
- Added strictest-wins central tool policy, schema validation/canonicalization before approval, approval policy validation, terminal policy denials, and aggregate execution limits.
- Preserved `Tool` struct layout for v1 source compatibility; direct `Tool.Call` remains available but deprecated because it bypasses toolbox enforcement.
- Verification passed:
  - `GOMODCACHE=/tmp/kisaragi-kit-go-mod-cache GOCACHE=/tmp/kisaragi-kit-go-build-cache go test ./...`
  - `GOMODCACHE=/tmp/kisaragi-kit-go-mod-cache GOCACHE=/tmp/kisaragi-kit-go-build-cache go test -race ./...`
  - `GOMODCACHE=/tmp/kisaragi-kit-go-mod-cache GOCACHE=/tmp/kisaragi-kit-go-build-cache go vet ./...`
  - `XDG_CACHE_HOME=/tmp/kisaragi-kit-staticcheck-cache GOMODCACHE=/tmp/kisaragi-kit-go-mod-cache GOCACHE=/tmp/kisaragi-kit-go-build-cache staticcheck ./...`

## Open Questions

None.

## Critique Pass

- Streaming leak gap resolved: guardrails inspect proposed accumulated content before observer delivery and suppress the crossing delta.
- Non-streaming/custom-provider gap resolved: final assistant checkpoint remains mandatory.
- Tool-argument and reasoning leak paths resolved: both are explicit matcher inputs.
- Security/persistence conflict resolved: safe partial prefix returns for UI reconciliation but blocked candidate never enters agent history.
- Genericity preserved: core defines allow/block pipeline; matcher lives in attachable subpackage.
- Compatibility risk contained: public APIs remain additive and direct `Tool.Call` is deprecated/documented rather than removed in v1.
- No remaining implementation decision or critique blocker.

## Assumptions

- Guardrails are trusted synchronous in-process code; asynchronous/network policy services can be called by applications through the supplied context.
- Defaults favor useful demonstration, not a claim of universal security efficacy. Applications tune matcher thresholds for their prompts and risk profile.
- `Message.Content` remains text-only; future multimodal parts require new guardrail channel semantics.

## Answer Log

- Selected recommended compatibility/tool defaults: additive v1, strictest-wins, abort on denial, stop remaining tool calls.
- Added generic every-role message guardrails with allow/block only.
- Added bundled attachable system-prompt leakage matcher and example.
- Preserved live deltas with pre-forward checks and mid-stream stop.
- Chose returned safe partial output without agent-history persistence.
