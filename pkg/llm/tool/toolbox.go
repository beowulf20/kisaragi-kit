package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

// Toolbox stores registered tools and calls them by name.
type Toolbox struct {
	tools map[string]Tool
}

// NewToolbox returns an empty toolbox.
func NewToolbox() *Toolbox {
	return &Toolbox{
		tools: make(map[string]Tool),
	}
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
	for _, tool := range tb.tools {
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
	tool, ok := tb.tools[name]
	if !ok {
		return nil, fmt.Errorf("tool %q is not registered", name)
	}

	args := json.RawMessage(arguments)
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	return tool.Call(ctx, args)
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
