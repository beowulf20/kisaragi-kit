package agent

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

// Agent wraps completion configuration with persistent message history.
type Agent struct {
	// Name identifies the agent and is used when exposing it as a tool.
	Name string
	// Description explains the agent's role when exposed as a tool.
	Description string
	// Hooks receives streaming content and tool execution events.
	Hooks Hooks
	llm.CompletionCallInput

	mu    sync.Mutex
	runMu sync.Mutex
}

// NewAgentInput configures a new Agent.
type NewAgentInput struct {
	// Name identifies the agent.
	Name string
	// Description explains the agent's role when exposed as a tool.
	Description string
	// SystemPrompt is prepended as a system message when set.
	SystemPrompt string
	// Hooks receives streaming content and tool execution events.
	Hooks Hooks
	// Config contains the completion client, model, tools, and initial messages.
	Config llm.CompletionCallInput
}

// Hooks contains optional callbacks emitted while an agent runs.
type Hooks struct {
	// OnContentDelta receives streamed assistant text chunks.
	OnContentDelta func(string)
	// OnReasoningDelta receives streamed assistant reasoning chunks when a provider exposes them.
	OnReasoningDelta func(string)
	// OnAssistantMessage runs after one provider response is returned and before requested tools run.
	OnAssistantMessage func(llm.AssistantMessageEvent)
	// OnEvent receives typed lifecycle events and may abort the agent run by returning an error.
	OnEvent func(llm.Event) error
	// OnGenerationStart runs before one provider generation attempt starts.
	OnGenerationStart func(llm.GenerationStartEvent)
	// OnGenerationEnd runs after one provider generation attempt ends.
	OnGenerationEnd func(llm.GenerationEndEvent)
	// OnUsage runs after a provider generation reports token usage.
	OnUsage func(llm.UsageEvent)
	// OnCallError runs after a provider chat call fails.
	OnCallError func(error)
	// OnToolCall runs before a requested tool is executed.
	OnToolCall func(llm.ToolCall)
	// OnToolError runs after a tool returns an error or invalid result.
	OnToolError func(llm.ToolCall, error)
	// OnToolResult runs after a tool returns or fails.
	OnToolResult func(llm.ToolCall)
}

// NewAgent validates input and returns an agent with copied initial messages.
func NewAgent(input NewAgentInput) (*Agent, error) {
	name := strings.TrimSpace(input.Name)
	if name == "" {
		return nil, errors.New("agent name cannot be empty")
	}

	config := input.Config
	config.Messages = append([]llm.Message(nil), input.Config.Messages...)
	if systemPrompt := strings.TrimSpace(input.SystemPrompt); systemPrompt != "" {
		config.Messages = append([]llm.Message{llm.NewSystemMessage(systemPrompt)}, config.Messages...)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}

	return &Agent{
		Name:                name,
		Description:         strings.TrimSpace(input.Description),
		Hooks:               input.Hooks,
		CompletionCallInput: config,
	}, nil
}

// AddMessage appends a message to the agent's persistent history.
func (a *Agent) AddMessage(message llm.Message) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.Messages = append(a.Messages, message)
}

// MessagesSnapshot returns a copy of the agent's persistent history.
func (a *Agent) MessagesSnapshot() []llm.Message {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]llm.Message(nil), a.Messages...)
}

// CallWithUserMessage appends a user message and runs the agent.
func (a *Agent) CallWithUserMessage(content string) (*llm.CompletionCallOutput, error) {
	return a.callWithUserMessage(content, context.Background(), false)
}

// CallWithUserMessageContext appends a user message and runs the agent with ctx.
func (a *Agent) CallWithUserMessageContext(ctx context.Context, content string) (*llm.CompletionCallOutput, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	return a.callWithUserMessage(content, ctx, true)
}

func (a *Agent) callWithUserMessage(content string, ctx context.Context, overrideContext bool) (*llm.CompletionCallOutput, error) {
	if a == nil {
		return nil, errors.New("agent is nil")
	}
	if strings.TrimSpace(content) == "" {
		return nil, errors.New("user message cannot be empty")
	}

	a.runMu.Lock()
	defer a.runMu.Unlock()

	a.mu.Lock()
	a.Messages = append(a.Messages, llm.NewUserMessage(content))
	input := a.completionInputLocked(a.Messages)
	if overrideContext {
		input.Context = ctx
	}
	a.mu.Unlock()

	return a.complete(input)
}

// Run continues the agent from its current persistent history.
func (a *Agent) Run() (*llm.CompletionCallOutput, error) {
	return a.run(context.Background(), false)
}

// RunContext continues the agent from its current persistent history with ctx.
func (a *Agent) RunContext(ctx context.Context) (*llm.CompletionCallOutput, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	return a.run(ctx, true)
}

func (a *Agent) run(ctx context.Context, overrideContext bool) (*llm.CompletionCallOutput, error) {
	if a == nil {
		return nil, errors.New("agent is nil")
	}

	a.runMu.Lock()
	defer a.runMu.Unlock()

	a.mu.Lock()
	input := a.completionInputLocked(a.Messages)
	if overrideContext {
		input.Context = ctx
	}
	a.mu.Unlock()

	return a.complete(input)
}

// RunWithTransientMessage runs the agent with an extra message that is not stored.
func (a *Agent) RunWithTransientMessage(message llm.Message) (*llm.CompletionCallOutput, error) {
	return a.runWithTransientMessage(message, context.Background(), false)
}

// RunWithTransientMessageContext runs the agent with an extra message and ctx.
func (a *Agent) RunWithTransientMessageContext(ctx context.Context, message llm.Message) (*llm.CompletionCallOutput, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	return a.runWithTransientMessage(message, ctx, true)
}

func (a *Agent) runWithTransientMessage(message llm.Message, ctx context.Context, overrideContext bool) (*llm.CompletionCallOutput, error) {
	if a == nil {
		return nil, errors.New("agent is nil")
	}

	a.runMu.Lock()
	defer a.runMu.Unlock()

	a.mu.Lock()
	messages := append(append([]llm.Message(nil), a.Messages...), message)
	input := a.completionInputLocked(messages)
	if overrideContext {
		input.Context = ctx
	}
	a.mu.Unlock()

	return a.complete(input)
}

func (a *Agent) complete(input llm.CompletionCallInput) (*llm.CompletionCallOutput, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}

	input.Hooks = llm.ChainCompletionHooks(input.Hooks, a.Hooks.completionHooks())

	output, err := llm.Completion(input)
	if err != nil {
		return output, err
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if len(output.Messages) > 0 {
		a.Messages = append(a.Messages, output.Messages...)
	} else if strings.TrimSpace(output.Content) != "" {
		a.Messages = append(a.Messages, llm.NewAssistantMessage(output.Content))
	}
	return output, nil
}

func (a *Agent) completionInputLocked(messages []llm.Message) llm.CompletionCallInput {
	input := a.CompletionCallInput
	input.Messages = append([]llm.Message(nil), messages...)
	return input
}

func (hooks Hooks) completionHooks() llm.CompletionHooks {
	return llm.CompletionHooks{
		OnContentDelta:     hooks.OnContentDelta,
		OnReasoningDelta:   hooks.OnReasoningDelta,
		OnAssistantMessage: hooks.OnAssistantMessage,
		OnEvent:            hooks.OnEvent,
		OnGenerationStart:  hooks.OnGenerationStart,
		OnGenerationEnd:    hooks.OnGenerationEnd,
		OnUsage:            hooks.OnUsage,
		OnCallError:        hooks.OnCallError,
		OnToolCall:         hooks.OnToolCall,
		OnToolError:        hooks.OnToolError,
		OnToolResult:       hooks.OnToolResult,
	}
}
