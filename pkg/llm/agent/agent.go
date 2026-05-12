package agent

import (
	"errors"
	"strings"

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
	a.Messages = append(a.Messages, message)
}

// MessagesSnapshot returns a copy of the agent's persistent history.
func (a *Agent) MessagesSnapshot() []llm.Message {
	return append([]llm.Message(nil), a.Messages...)
}

// CallWithUserMessage appends a user message and runs the agent.
func (a *Agent) CallWithUserMessage(content string) (*llm.CompletionCallOutput, error) {
	if strings.TrimSpace(content) == "" {
		return nil, errors.New("user message cannot be empty")
	}
	a.AddMessage(llm.NewUserMessage(content))
	return a.call()
}

// Run continues the agent from its current persistent history.
func (a *Agent) Run() (*llm.CompletionCallOutput, error) {
	return a.call()
}

// RunWithTransientMessage runs the agent with an extra message that is not stored.
func (a *Agent) RunWithTransientMessage(message llm.Message) (*llm.CompletionCallOutput, error) {
	if a == nil {
		return nil, errors.New("agent is nil")
	}
	input := a.CompletionCallInput
	input.Messages = append(append([]llm.Message(nil), a.Messages...), message)
	return a.complete(input)
}

func (a *Agent) call() (*llm.CompletionCallOutput, error) {
	if a == nil {
		return nil, errors.New("agent is nil")
	}
	return a.complete(a.CompletionCallInput)
}

func (a *Agent) complete(input llm.CompletionCallInput) (*llm.CompletionCallOutput, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}

	input.Hooks = llm.CompletionHooks{
		OnContentDelta: a.Hooks.OnContentDelta,
		OnCallError:    a.Hooks.OnCallError,
		OnToolCall:     a.Hooks.OnToolCall,
		OnToolError:    a.Hooks.OnToolError,
		OnToolResult:   a.Hooks.OnToolResult,
	}

	output, err := llm.Completion(input)
	if err != nil {
		return nil, err
	}

	if len(output.Messages) > 0 {
		a.Messages = append(a.Messages, output.Messages...)
	} else if strings.TrimSpace(output.Content) != "" {
		a.Messages = append(a.Messages, llm.NewAssistantMessage(output.Content))
	}
	return output, nil
}
