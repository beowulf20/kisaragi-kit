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
	policyHook   ToolPolicyHook
}

// ToolboxOption customizes a toolbox.
type ToolboxOption func(*Toolbox)

// WithApprovalHook installs a hook that can approve or deny tool calls.
func WithApprovalHook(hook ApprovalHook) ToolboxOption {
	return func(tb *Toolbox) {
		tb.approvalHook = hook
	}
}

// WithToolPolicyHook installs an application-owned policy evaluated for every tool call.
func WithToolPolicyHook(hook ToolPolicyHook) ToolboxOption {
	return func(tb *Toolbox) {
		tb.policyHook = hook
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
	if err := tool.Approval.validate(); err != nil {
		return fmt.Errorf("tool %q: %w", tool.Name, err)
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

// ToolCallRequest carries runtime context for protected tool execution.
type ToolCallRequest struct {
	ID        string
	Name      string
	Arguments string
	Round     int
	Model     string
}

// CallOutput contains a tool result plus runtime metadata.
type CallOutput struct {
	Result   *string
	Approval *ApprovalRecord
}

// CallWithInfo executes a registered tool and returns runtime metadata.
func (tb *Toolbox) CallWithInfo(ctx context.Context, name string, arguments string) (CallOutput, error) {
	return tb.CallWithRequest(ctx, ToolCallRequest{Name: name, Arguments: arguments})
}

// CallWithRequest validates, authorizes, approves, and executes one registered tool.
func (tb *Toolbox) CallWithRequest(ctx context.Context, request ToolCallRequest) (CallOutput, error) {
	tool, ok := tb.tools[request.Name]
	if !ok {
		return CallOutput{}, fmt.Errorf("tool %q is not registered", request.Name)
	}

	args := json.RawMessage(request.Arguments)
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	prepared, err := tool.prepareArguments(args)
	if err != nil {
		return CallOutput{}, err
	}
	policyAction, policyReason, err := evaluateToolPolicy(ctx, tb.policyHook, ToolPolicyRequest{
		ToolCallID:  request.ID,
		ToolName:    tool.Name,
		Description: tool.Description,
		Arguments:   append(json.RawMessage(nil), prepared...),
		Policy:      tool.Approval,
		Round:       request.Round,
		Model:       request.Model,
	})
	if err != nil {
		return CallOutput{}, err
	}
	declaredApproval := tool.Approval.requiresApproval()
	approval, err := approveToolCall(
		ctx,
		tb.approvalHook,
		request,
		tool,
		prepared,
		policyAction == ToolPolicyRequireApproval,
		policyAction == ToolPolicyRequireApproval && !declaredApproval,
		policyReason,
	)
	if err != nil {
		return CallOutput{Approval: approval}, err
	}
	result, err := callToolSafely(ctx, tool, prepared)
	return CallOutput{Result: result, Approval: approval}, err
}

func callToolSafely(ctx context.Context, tool Tool, arguments json.RawMessage) (result *string, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			result = nil
			err = fmt.Errorf("tool %q panicked: %v", tool.Name, recovered)
		}
	}()
	return tool.Call(ctx, arguments)
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
