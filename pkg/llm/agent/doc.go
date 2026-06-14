// Package agent builds stateful LLM agents on top of package llm.
//
// Agents own message history, can run with persistent or transient user input,
// and can be exposed as tools so one agent can delegate work to another. Agent
// methods serialize runs and protect history; callers should not mutate
// exported fields concurrently with a run.
package agent
