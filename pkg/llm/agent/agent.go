package agent

import (
	"errors"
	"strings"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

type Agent struct {
	Name        string
	Description string
	Hooks       Hooks
	llm.CompletionCallInput
}

type NewAgentInput struct {
	Name         string
	Description  string
	SystemPrompt string
	Hooks        Hooks
	Config       llm.CompletionCallInput
}

type Hooks struct {
	OnContentDelta func(string)
	OnToolCall     func(llm.ToolCall)
	OnToolResult   func(llm.ToolCall)
}

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

func (a *Agent) AddMessage(message llm.Message) {
	a.Messages = append(a.Messages, message)
}

func (a *Agent) MessagesSnapshot() []llm.Message {
	return append([]llm.Message(nil), a.Messages...)
}

func (a *Agent) CallWithUserMessage(content string) (*llm.CompletionCallOutput, error) {
	if strings.TrimSpace(content) == "" {
		return nil, errors.New("user message cannot be empty")
	}
	a.AddMessage(llm.NewUserMessage(content))
	return a.call()
}

func (a *Agent) Run() (*llm.CompletionCallOutput, error) {
	return a.call()
}

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
		OnToolCall:     a.Hooks.OnToolCall,
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
