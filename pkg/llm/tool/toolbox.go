package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
)

// Toolbox stores registered tools and calls them by name.
type Toolbox struct {
	tools        map[string]Tool
	approvalHook ApprovalHook
}

// ToolboxOption customizes a toolbox.
type ToolboxOption func(*Toolbox)

// WithApprovalHook installs a hook that can approve or deny tool calls.
func WithApprovalHook(hook ApprovalHook) ToolboxOption {
	return func(tb *Toolbox) {
		tb.approvalHook = hook
	}
}

// NewToolbox returns an empty toolbox.
func NewToolbox(options ...ToolboxOption) *Toolbox {
	tb := &Toolbox{
		tools: make(map[string]Tool),
	}
	for _, option := range options {
		option(tb)
	}
	return tb
}

// RegisterTool adds a tool to the toolbox.
func (tb *Toolbox) RegisterTool(tool Tool) error {
	if tool.Name == "" {
		return errors.New("tool name cannot be empty")
	}
	if tool.Call == nil {
		return fmt.Errorf("tool %q has no call handler", tool.Name)
	}
	if tb.tools == nil {
		tb.tools = make(map[string]Tool)
	}
	if _, ok := tb.tools[tool.Name]; ok {
		return fmt.Errorf("tool %q already registered", tool.Name)
	}

	tb.tools[tool.Name] = tool
	return nil
}

// ChatTools returns provider-neutral tool definitions for chat requests.
func (tb *Toolbox) ChatTools() []ChatTool {
	tools := make([]ChatTool, 0, len(tb.tools))
	names := make([]string, 0, len(tb.tools))
	for name := range tb.tools {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		tool := tb.tools[name]
		tools = append(tools, ChatTool{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
		})
	}
	return tools
}

// Call executes a registered tool with a JSON argument payload.
func (tb *Toolbox) Call(ctx context.Context, name string, arguments string) (*string, error) {
	output, err := tb.CallWithInfo(ctx, name, arguments)
	return output.Result, err
}

// CallOutput contains a tool result plus runtime metadata.
type CallOutput struct {
	Result   *string
	Approval *ApprovalRecord
}

// CallWithInfo executes a registered tool and returns runtime metadata.
func (tb *Toolbox) CallWithInfo(ctx context.Context, name string, arguments string) (CallOutput, error) {
	tool, ok := tb.tools[name]
	if !ok {
		return CallOutput{}, fmt.Errorf("tool %q is not registered", name)
	}

	args := json.RawMessage(arguments)
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	approval, err := approveToolCall(ctx, tb.approvalHook, tool, args)
	if err != nil {
		return CallOutput{Approval: approval}, err
	}
	result, err := tool.Call(ctx, args)
	return CallOutput{Result: result, Approval: approval}, err
}

// ChatTool is the provider-neutral schema exposed to chat clients.
type ChatTool struct {
	// Name is the function name exposed to the model.
	Name string
	// Description explains what the tool does.
	Description string
	// Parameters is the JSON schema for the tool input.
	Parameters map[string]any
}
