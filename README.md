# Kisaragi Kit

![Kisaragi Kit mascot](assets/kisaragi-kit-logo-source.png)

Small Go framework for LLM apps, with an OpenAI-compatible adapter. It gives this repo a reusable core for:

- streaming chat completions
- typed tool registration
- JSON schema generation from Go structs
- tool-call execution loops
- stateful agents
- agents exposed as tools for delegation

## Name

Kisaragi Kit is a nod to the Kisaragi Corporation from *Combatants Will Be Dispatched!*: a small, reusable kit for dispatching agents, wiring specialist helpers together, and turning agents into callable tools.

## Install

```bash
go get github.com/beowulf20/kisaragi-kit
```

## Packages

| Package | Purpose |
| --- | --- |
| `pkg/llm` | Provider-neutral message types, completion loop, hook events, tool-call transcript handling. |
| `pkg/llm/tool` | Generic `NewTool[I, O]` helper, toolbox registry, JSON schema generation, typed input decoding. |
| `pkg/llm/agent` | Stateful agents with persistent messages, transient runs, streaming hooks, and `AsTool()` delegation. |
| `pkg/llm/provider/openai` | OpenAI-compatible client adapter, streaming conversion, model listing, and tool/message translation. |

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/agent"
	openaiadapter "github.com/beowulf20/kisaragi-kit/pkg/llm/provider/openai"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

type weatherInput struct {
	City string `json:"city" description:"City to check"`
}

type weatherOutput struct {
	Summary string `json:"summary"`
}

func main() {
	client, _, err := openaiadapter.NewClient(openaiadapter.ClientConfig{
		BaseURL: "https://api.openai.com/v1",
		APIKey:  os.Getenv("OPENAI_API_KEY"),
		Timeout: 60 * time.Second,
	})
	if err != nil {
		log.Fatal(err)
	}

	tools := llmtool.NewToolbox()
	err = tools.RegisterTool(llmtool.NewTool("weather", "Gets current weather.", func(_ context.Context, input weatherInput) (weatherOutput, error) {
		return weatherOutput{Summary: "clear skies in " + input.City}, nil
	}))
	if err != nil {
		log.Fatal(err)
	}

	assistant, err := agent.NewAgent(agent.NewAgentInput{
		Name:         "assistant",
		SystemPrompt: "Answer briefly. Use tools when they help.",
		Config: llm.CompletionCallInput{
			Client: client,
			Model:  "gpt-4o-mini",
			Tools:  *tools,
			// Optional safety limits; zero values use package defaults.
			MaxToolCallRounds: llm.DefaultMaxToolCallRounds,
			MaxToolErrorLength: llm.DefaultMaxToolErrorLength,
			// Optional provider retry limit; nil uses package default.
			ProviderErrorRetries: nil,
		},
		Hooks: agent.Hooks{
			OnContentDelta: func(delta string) { fmt.Print(delta) },
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	if _, err := assistant.CallWithUserMessage("What is the weather in Curitiba?"); err != nil {
		log.Fatal(err)
	}
}
```

Run the included example:

```bash
OPENAI_API_KEY=... go run ./examples/basic
```

For OpenAI-compatible local servers:

```bash
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=local OPENAI_MODEL=llama3.1 go run ./examples/basic
```

Run the human-approval example:

```bash
OPENAI_API_KEY=... go run ./examples/approval
```

## Core Concepts

### Messages

Use constructors from `pkg/llm`:

- `NewSystemMessage`
- `NewUserMessage`
- `NewAssistantMessage`
- `NewAssistantToolCallMessage`
- `NewToolMessage`

The completion loop returns appended assistant/tool messages through `CompletionCallOutput.Messages`, so callers can persist conversation history.

### Tools

Tools are normal Go functions:

```go
tool := llmtool.NewTool("lookup", "Looks up a record.", func(ctx context.Context, input lookupInput) (lookupOutput, error) {
	return lookupOutput{}, nil
})
```

Input structs become provider-neutral JSON schemas. Public fields are included, `json:"-"` fields are ignored, pointer fields and `omitempty` fields are optional, and `description` tags become schema descriptions.

Tool errors are sent back to the model as tool result messages, so the next turn can repair bad arguments or choose another path. Override the feedback with `ToolErrorInterceptor`:

```go
ToolErrorInterceptor: func(ctx llm.ToolErrorContext) llm.ToolErrorDecision {
	return llm.ToolErrorDecision{
		Feedback: `{"error":"missing city","retryable":true,"hint":"Call weather with a city."}`,
	}
},
```

Tool approval is declared on the tool and enforced by the toolbox before the Go handler runs:

```go
toolbox := llmtool.NewToolbox(llmtool.WithApprovalHook(llmtool.NewStdioApprovalHook(os.Stdin, os.Stdout)))
tool := llmtool.NewTool("delete_record", "Deletes a record.", deleteRecord, llmtool.WithApproval(llmtool.ApprovalPolicy{
	Mode:        llmtool.ApprovalAlways,
	Risk:        llmtool.RiskHigh,
	Preview:     llmtool.PreviewPayload,
	Description: "Delete one record by ID.",
}))
_ = toolbox.RegisterTool(tool)
```

Custom approval hooks can show diffs, command previews, UI prompts, or audit-log entries. If a tool requires approval and no hook is installed, the call fails with `ErrApprovalDenied`.

Agents persist approval decisions when `CompletionCallInput.ApprovalDecisionMessages` enables accepted or rejected transcript messages.

### Agents

Agents wrap completion config plus message history:

```go
assistant, err := agent.NewAgent(agent.NewAgentInput{
	Name:         "researcher",
	SystemPrompt: "Be precise.",
	Config: llm.CompletionCallInput{
		Client: client,
		Model:  "gpt-4o-mini",
	},
})
```

Use `CallWithUserMessage` for normal conversation, `Run` to continue from existing state, and `RunWithTransientMessage` when extra context should be sent once without being stored.

### Agent Delegation

Any agent can become a tool:

```go
toolbox := llmtool.NewToolbox()
_ = toolbox.RegisterTool(researchAgent.AsTool())
```

This lets a coordinator agent call specialist agents using the same tool loop.

## Hooks

`CompletionHooks` and `agent.Hooks` support:

- `OnContentDelta` for streaming text
- `OnCallError` after a provider chat call fails
- `OnToolCall` before a tool runs
- `OnToolError` after a tool returns an error or invalid result
- `OnToolResult` after a tool returns or fails
- `llmtool.ApprovalHook` before approved tool execution

## Tests

```bash
go test ./...
```

The test suite covers schema generation, typed tool calls, streaming content, tool-call messages, agent history, transient messages, and agent-as-tool behavior.

## Design Notes

- The framework core is provider-neutral; the OpenAI adapter targets OpenAI-compatible chat completion APIs.
- Tool execution is capped by `CompletionCallInput.MaxToolCallRounds` to prevent infinite loops.
- Tool approval is metadata on `llmtool.Tool`; handlers stay ordinary Go functions, and `Toolbox.Call` owns enforcement.
- Approval accept/reject transcript messages are opt-in through `CompletionCallInput.ApprovalDecisionMessages`.
- Tool errors are returned to the model as JSON error payloads capped by `CompletionCallInput.MaxToolErrorLength`.
- Tool error feedback can be rewritten or aborted by `CompletionCallInput.ToolErrorInterceptor`.
- Provider completion errors are retried by `CompletionCallInput.ProviderErrorRetries`; nil uses `DefaultProviderErrorRetries`.
- The public API stays small: client setup, messages, completions, tools, and agents.
