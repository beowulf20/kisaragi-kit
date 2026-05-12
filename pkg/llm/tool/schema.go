package tool

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"time"
)

// JSONSchemaFor returns an object JSON schema for values of type T.
func JSONSchemaFor[T any]() map[string]any {
	t := reflect.TypeOf((*T)(nil)).Elem()
	schema := jsonSchemaForInputType(t)
	if schemaType, _ := schema["type"].(string); schemaType == "object" {
		return schema
	}

	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": schema,
		},
		"required":             []string{"value"},
		"additionalProperties": false,
	}
}

func decodeToolInput[T any](arguments json.RawMessage) (T, error) {
	var input T
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}

	decoder := json.NewDecoder(bytes.NewReader(arguments))
	if isStructType(reflect.TypeOf((*T)(nil)).Elem()) {
		decoder.DisallowUnknownFields()
	}
	if err := decoder.Decode(&input); err != nil {
		return input, err
	}
	return input, nil
}

func jsonSchemaForInputType(t reflect.Type) map[string]any {
	t = dereferenceType(t)
	if t.Kind() == reflect.Struct && t.NumField() == 0 {
		return emptyObjectSchema()
	}
	return jsonSchemaForType(t)
}

func jsonSchemaForType(t reflect.Type) map[string]any {
	t = dereferenceType(t)
	if isTimeType(t) {
		return map[string]any{"type": "string", "format": "date-time"}
	}

	switch t.Kind() {
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Slice, reflect.Array:
		return map[string]any{
			"type":  "array",
			"items": jsonSchemaForType(t.Elem()),
		}
	case reflect.Map:
		schema := map[string]any{"type": "object"}
		if t.Key().Kind() == reflect.String {
			schema["additionalProperties"] = jsonSchemaForType(t.Elem())
		}
		return schema
	case reflect.Struct:
		return jsonSchemaForStruct(t)
	default:
		return map[string]any{"type": "object"}
	}
}

func jsonSchemaForStruct(t reflect.Type) map[string]any {
	properties := make(map[string]any)
	required := make([]string, 0, t.NumField())

	for i := range t.NumField() {
		field := t.Field(i)
		if field.PkgPath != "" || field.Anonymous {
			continue
		}

		name, omitempty, ok := jsonFieldName(field)
		if !ok {
			continue
		}

		property := jsonSchemaForType(field.Type)
		if description := field.Tag.Get("description"); description != "" {
			property["description"] = description
		}
		properties[name] = property

		if isRequiredField(field.Type, omitempty) {
			required = append(required, name)
		}
	}

	schema := map[string]any{
		"type":                 "object",
		"properties":           properties,
		"additionalProperties": false,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

func emptyObjectSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"properties":           map[string]any{},
		"additionalProperties": false,
	}
}

func jsonFieldName(field reflect.StructField) (string, bool, bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, false
	}

	parts := strings.Split(tag, ",")
	name := parts[0]
	if name == "" {
		name = field.Name
	}

	omitempty := false
	for _, option := range parts[1:] {
		if option == "omitempty" || option == "omitzero" {
			omitempty = true
			break
		}
	}
	return name, omitempty, true
}

func dereferenceType(t reflect.Type) reflect.Type {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t
}

func isStructType(t reflect.Type) bool {
	t = dereferenceType(t)
	return t.Kind() == reflect.Struct && !isTimeType(t)
}

func isRequiredField(t reflect.Type, omitempty bool) bool {
	return !omitempty && t.Kind() != reflect.Pointer
}

func isTimeType(t reflect.Type) bool {
	return t == reflect.TypeOf(time.Time{})
}
