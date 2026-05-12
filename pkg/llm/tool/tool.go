package tool

import (
	"context"
	"encoding/json"
	"fmt"
)

// Tool describes a callable LLM tool and its JSON schema.
type Tool struct {
	// Name is the function name exposed to the model.
	Name string
	// Description explains what the tool does.
	Description string
	// Parameters is the JSON schema for the tool input.
	Parameters map[string]any
	// Call decodes raw JSON arguments and returns a JSON result.
	Call func(context.Context, json.RawMessage) (*string, error)
}

// NewTool wraps a typed Go function as a Tool with inferred input schema.
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
