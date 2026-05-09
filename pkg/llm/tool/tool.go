package tool

import (
	"context"
	"encoding/json"
	"fmt"
)

type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any
	Call        func(context.Context, json.RawMessage) (*string, error)
}

func NewTool[I, O any](name string, description string, call func(context.Context, I) (O, error)) Tool {
	return Tool{
		Name:        name,
		Description: description,
		Parameters:  JSONSchemaFor[I](),
		Call: func(ctx context.Context, arguments json.RawMessage) (*string, error) {
			input, err := decodeToolInput[I](arguments)
			if err != nil {
				return nil, fmt.Errorf("parse %s arguments: %w", name, err)
			}

			output, err := call(ctx, input)
			if err != nil {
				return nil, err
			}

			data, err := json.Marshal(output)
			if err != nil {
				return nil, fmt.Errorf("marshal %s result: %w", name, err)
			}
			result := string(data)
			return &result, nil
		},
	}
}
