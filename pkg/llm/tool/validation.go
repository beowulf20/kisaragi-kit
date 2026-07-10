package tool

import (
	"encoding/json"
	"fmt"
	"math"
)

func validateValueAgainstSchema(schema map[string]any, value any, path string) error {
	if schema == nil {
		return nil
	}
	if value == nil {
		return nil
	}
	switch schema["type"] {
	case "object":
		return validateObjectAgainstSchema(schema, value, path)
	case "array":
		values, ok := value.([]any)
		if !ok {
			return typeMismatch(path, "array")
		}
		items, _ := schema["items"].(map[string]any)
		for index, item := range values {
			if err := validateValueAgainstSchema(items, item, fmt.Sprintf("%s[%d]", displayPath(path), index)); err != nil {
				return err
			}
		}
	case "string":
		if _, ok := value.(string); !ok {
			return typeMismatch(path, "string")
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return typeMismatch(path, "boolean")
		}
	case "number":
		if _, ok := value.(json.Number); !ok {
			return typeMismatch(path, "number")
		}
	case "integer":
		number, ok := value.(json.Number)
		parsed, parseErr := number.Float64()
		if !ok || parseErr != nil || math.Trunc(parsed) != parsed {
			return typeMismatch(path, "integer")
		}
	}
	return nil
}

func validateObjectAgainstSchema(schema map[string]any, value any, path string) error {
	object, ok := value.(map[string]any)
	if !ok {
		return typeMismatch(path, "object")
	}
	for _, name := range requiredPropertyNames(schema["required"]) {
		if _, exists := object[name]; !exists {
			return fmt.Errorf("missing required argument %s", joinJSONPath(path, name))
		}
	}
	properties, _ := schema["properties"].(map[string]any)
	additional, hasAdditional := schema["additionalProperties"]
	for name, childValue := range object {
		childPath := joinJSONPath(path, name)
		if childSchema, ok := properties[name].(map[string]any); ok {
			if err := validateValueAgainstSchema(childSchema, childValue, childPath); err != nil {
				return err
			}
			continue
		}
		switch typed := additional.(type) {
		case bool:
			if hasAdditional && !typed {
				return fmt.Errorf("unknown argument %s", childPath)
			}
		case map[string]any:
			if err := validateValueAgainstSchema(typed, childValue, childPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func requiredPropertyNames(value any) []string {
	switch typed := value.(type) {
	case []string:
		return typed
	case []any:
		names := make([]string, 0, len(typed))
		for _, item := range typed {
			if name, ok := item.(string); ok {
				names = append(names, name)
			}
		}
		return names
	default:
		return nil
	}
}

func typeMismatch(path string, expected string) error {
	return fmt.Errorf("argument %s must be %s", displayPath(path), expected)
}

func displayPath(path string) string {
	if path == "" {
		return "value"
	}
	return path
}

func joinJSONPath(path string, name string) string {
	if path == "" {
		return name
	}
	return path + "." + name
}
