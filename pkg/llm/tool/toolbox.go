package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
)

type Toolbox struct {
	tools map[string]Tool
}

func NewToolbox() *Toolbox {
	return &Toolbox{
		tools: make(map[string]Tool),
	}
}

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

type ChatTool struct {
	Name        string
	Description string
	Parameters  map[string]any
}
