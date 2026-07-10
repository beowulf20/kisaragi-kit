// Package llm provides a small framework for building streaming LLM agents.
//
// It defines provider-neutral chat interfaces, tracks message history, executes
// typed tools, applies ordered message guardrails and execution budgets, emits
// streaming hooks, and returns tool-call transcripts that can be persisted by
// callers.
package llm
