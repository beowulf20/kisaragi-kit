package tool

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestJSONSchemaForEmptyInput(t *testing.T) {
	type input struct{}

	schema := JSONSchemaFor[input]()

	if schema["type"] != "object" {
		t.Fatalf("type = %v, want object", schema["type"])
	}
	if additional := schema["additionalProperties"]; additional != false {
		t.Fatalf("additionalProperties = %v, want false", additional)
	}

	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties has type %T, want map[string]any", schema["properties"])
	}
	if len(properties) != 0 {
		t.Fatalf("properties = %v, want empty", properties)
	}
}

func TestJSONSchemaForStructInput(t *testing.T) {
	type input struct {
		Name  string `json:"name" description:"Name to greet"`
		Count int    `json:"count,omitempty"`
		Skip  string `json:"-"`
	}

	schema := JSONSchemaFor[input]()
	properties := schema["properties"].(map[string]any)

	name := properties["name"].(map[string]any)
	if name["type"] != "string" {
		t.Fatalf("name.type = %v, want string", name["type"])
	}
	if name["description"] != "Name to greet" {
		t.Fatalf("name.description = %v, want description tag", name["description"])
	}

	count := properties["count"].(map[string]any)
	if count["type"] != "integer" {
		t.Fatalf("count.type = %v, want integer", count["type"])
	}
	if _, ok := properties["Skip"]; ok {
		t.Fatal("json:\"-\" field should not appear in schema")
	}

	required := schema["required"].([]string)
	if len(required) != 1 || required[0] != "name" {
		t.Fatalf("required = %v, want [name]", required)
	}
}

func TestJSONSchemaForPointerFieldsMarksThemOptional(t *testing.T) {
	type input struct {
		Name     string  `json:"name"`
		Nickname *string `json:"nickname" description:"Optional nickname"`
		Count    *int    `json:"count"`
	}

	schema := JSONSchemaFor[input]()
	properties := schema["properties"].(map[string]any)

	nickname := properties["nickname"].(map[string]any)
	if nickname["type"] != "string" {
		t.Fatalf("nickname.type = %v, want string", nickname["type"])
	}
	if nickname["description"] != "Optional nickname" {
		t.Fatalf("nickname.description = %v, want description tag", nickname["description"])
	}

	count := properties["count"].(map[string]any)
	if count["type"] != "integer" {
		t.Fatalf("count.type = %v, want integer", count["type"])
	}

	required := schema["required"].([]string)
	if len(required) != 1 || required[0] != "name" {
		t.Fatalf("required = %v, want only [name]", required)
	}
}

func TestJSONSchemaForPointerInput(t *testing.T) {
	type input struct {
		Name string `json:"name"`
	}

	schema := JSONSchemaFor[*input]()
	properties := schema["properties"].(map[string]any)

	name := properties["name"].(map[string]any)
	if name["type"] != "string" {
		t.Fatalf("name.type = %v, want string", name["type"])
	}

	required := schema["required"].([]string)
	if len(required) != 1 || required[0] != "name" {
		t.Fatalf("required = %v, want [name]", required)
	}
}

func TestJSONSchemaForNilPointerInputIsEmptySchema(t *testing.T) {
	type input struct{}

	schema := JSONSchemaFor[*input]()

	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties has type %T, want map[string]any", schema["properties"])
	}
	if len(properties) != 0 {
		t.Fatalf("properties = %v, want empty", properties)
	}
	if additional := schema["additionalProperties"]; additional != false {
		t.Fatalf("additionalProperties = %v, want false", additional)
	}
}

func TestToolboxCallUsesPointerInput(t *testing.T) {
	type input struct {
		Name string `json:"name"`
	}
	type output struct {
		Greeting string `json:"greeting"`
	}

	toolbox := NewToolbox()
	err := toolbox.RegisterTool(NewTool("greet_ptr", "Greets a person.", func(_ context.Context, input *input) (output, error) {
		if input == nil {
			t.Fatal("input is nil")
		}
		return output{Greeting: "hello " + input.Name}, nil
	}))
	if err != nil {
		t.Fatal(err)
	}

	result, err := toolbox.Call(context.Background(), "greet_ptr", `{"name":"Ada"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("result is nil")
	}

	var decoded output
	if err := json.Unmarshal([]byte(*result), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Greeting != "hello Ada" {
		t.Fatalf("greeting = %q, want hello Ada", decoded.Greeting)
	}
}

func TestToolboxCallUsesTypedInput(t *testing.T) {
	type input struct {
		Name string `json:"name"`
	}
	type output struct {
		Greeting string `json:"greeting"`
	}

	toolbox := NewToolbox()
	err := toolbox.RegisterTool(NewTool("greet", "Greets a person.", func(_ context.Context, input input) (output, error) {
		return output{Greeting: "hello " + input.Name}, nil
	}))
	if err != nil {
		t.Fatal(err)
	}

	result, err := toolbox.Call(context.Background(), "greet", `{"name":"Ada"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("result is nil")
	}

	var decoded output
	if err := json.Unmarshal([]byte(*result), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Greeting != "hello Ada" {
		t.Fatalf("greeting = %q, want hello Ada", decoded.Greeting)
	}
}

func TestToolboxCallLeavesMissingPointerInputNil(t *testing.T) {
	type input struct {
		Name     string  `json:"name"`
		Nickname *string `json:"nickname"`
	}
	type output struct {
		NicknameWasNil bool `json:"nickname_was_nil"`
	}

	toolbox := NewToolbox()
	err := toolbox.RegisterTool(NewTool("check_optional", "Checks optional pointer input.", func(_ context.Context, input input) (output, error) {
		return output{NicknameWasNil: input.Nickname == nil}, nil
	}))
	if err != nil {
		t.Fatal(err)
	}

	result, err := toolbox.Call(context.Background(), "check_optional", `{"name":"Ada"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("result is nil")
	}

	var decoded output
	if err := json.Unmarshal([]byte(*result), &decoded); err != nil {
		t.Fatal(err)
	}
	if !decoded.NicknameWasNil {
		t.Fatal("missing optional pointer input should decode as nil")
	}
}

func TestToolboxCallReturnsNilResultOnError(t *testing.T) {
	type input struct{}
	type output struct{}

	wantErr := errors.New("boom")
	toolbox := NewToolbox()
	err := toolbox.RegisterTool(NewTool("fail", "Fails.", func(_ context.Context, _ input) (output, error) {
		return output{}, wantErr
	}))
	if err != nil {
		t.Fatal(err)
	}

	result, err := toolbox.Call(context.Background(), "fail", `{}`)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
	if result != nil {
		t.Fatalf("result = %q, want nil", *result)
	}
}
