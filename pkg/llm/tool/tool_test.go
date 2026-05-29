package tool

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
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

func TestToolboxCallRequiresApprovalBeforeHandlerRuns(t *testing.T) {
	type input struct {
		Name string `json:"name"`
	}
	type output struct {
		Greeting string `json:"greeting"`
	}

	called := false
	toolbox := NewToolbox(WithApprovalHook(func(_ context.Context, request ApprovalRequest) (ApprovalDecision, error) {
		if request.ToolName != "greet" {
			t.Fatalf("tool name = %q, want greet", request.ToolName)
		}
		if string(request.Arguments) != `{"name":"Ada"}` {
			t.Fatalf("arguments = %s, want Ada payload", string(request.Arguments))
		}
		return ApprovalDecision{Approved: false, Reason: "operator said no"}, nil
	}))
	err := toolbox.RegisterTool(NewTool("greet", "Greets a person.", func(_ context.Context, input input) (output, error) {
		called = true
		return output{Greeting: "hello " + input.Name}, nil
	}, WithApproval(ApprovalPolicy{
		Mode: ApprovalAlways,
		Risk: RiskMedium,
	})))
	if err != nil {
		t.Fatal(err)
	}

	result, err := toolbox.Call(context.Background(), "greet", `{"name":"Ada"}`)
	if !errors.Is(err, ErrApprovalDenied) {
		t.Fatalf("err = %v, want ErrApprovalDenied", err)
	}
	if result != nil {
		t.Fatalf("result = %q, want nil", *result)
	}
	if called {
		t.Fatal("handler ran before approval")
	}
}

func TestToolboxCallRunsWhenApproved(t *testing.T) {
	type input struct {
		Name string `json:"name"`
	}
	type output struct {
		Greeting string `json:"greeting"`
	}

	toolbox := NewToolbox(WithApprovalHook(func(context.Context, ApprovalRequest) (ApprovalDecision, error) {
		return ApprovalDecision{Approved: true}, nil
	}))
	err := toolbox.RegisterTool(NewTool("greet", "Greets a person.", func(_ context.Context, input input) (output, error) {
		return output{Greeting: "hello " + input.Name}, nil
	}, WithApproval(ApprovalPolicy{
		Mode: ApprovalAlways,
		Risk: RiskMedium,
	})))
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

func TestToolboxCallWithInfoReturnsApprovalRecord(t *testing.T) {
	type input struct {
		Name string `json:"name"`
	}
	type output struct {
		Greeting string `json:"greeting"`
	}

	toolbox := NewToolbox(WithApprovalHook(func(context.Context, ApprovalRequest) (ApprovalDecision, error) {
		return ApprovalDecision{Approved: true, Reason: "approved by test"}, nil
	}))
	err := toolbox.RegisterTool(NewTool("greet", "Greets a person.", func(_ context.Context, input input) (output, error) {
		return output{Greeting: "hello " + input.Name}, nil
	}, WithApproval(ApprovalPolicy{
		Mode: ApprovalAlways,
		Risk: RiskMedium,
	})))
	if err != nil {
		t.Fatal(err)
	}

	result, err := toolbox.CallWithInfo(context.Background(), "greet", `{"name":"Ada"}`)
	if err != nil {
		t.Fatal(err)
	}
	if result.Approval == nil {
		t.Fatal("approval record is nil")
	}
	if !result.Approval.Approved || result.Approval.Reason != "approved by test" {
		t.Fatalf("approval = %#v, want approved record", result.Approval)
	}
	if result.Result == nil {
		t.Fatal("result is nil")
	}
}

func TestToolboxCallErrorsWhenApprovalHookMissing(t *testing.T) {
	type input struct{}
	type output struct{}

	toolbox := NewToolbox()
	err := toolbox.RegisterTool(NewTool("risky", "Needs approval.", func(context.Context, input) (output, error) {
		return output{}, nil
	}, WithApproval(ApprovalPolicy{
		Mode: ApprovalAlways,
		Risk: RiskHigh,
	})))
	if err != nil {
		t.Fatal(err)
	}

	_, err = toolbox.Call(context.Background(), "risky", `{}`)
	if !errors.Is(err, ErrApprovalDenied) {
		t.Fatalf("err = %v, want ErrApprovalDenied", err)
	}
}

func TestApprovalOnRiskSkipsLowRisk(t *testing.T) {
	type input struct{}
	type output struct{}

	approvalCalled := false
	toolbox := NewToolbox(WithApprovalHook(func(context.Context, ApprovalRequest) (ApprovalDecision, error) {
		approvalCalled = true
		return ApprovalDecision{}, nil
	}))
	err := toolbox.RegisterTool(NewTool("safe", "Low risk.", func(context.Context, input) (output, error) {
		return output{}, nil
	}, WithApproval(ApprovalPolicy{
		Mode: ApprovalOnRisk,
		Risk: RiskLow,
	})))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := toolbox.Call(context.Background(), "safe", `{}`); err != nil {
		t.Fatal(err)
	}
	if approvalCalled {
		t.Fatal("approval hook called for low-risk tool")
	}
}

func TestStdioApprovalHookApprovesYes(t *testing.T) {
	var out strings.Builder
	hook := NewStdioApprovalHook(strings.NewReader("yes\n"), &out)

	decision, err := hook(context.Background(), ApprovalRequest{
		ToolName:  "weather",
		Arguments: json.RawMessage(`{"city":"Curitiba"}`),
		Policy: ApprovalPolicy{
			Risk:        RiskMedium,
			Preview:     PreviewPayload,
			Description: "Get current weather.",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !decision.Approved {
		t.Fatal("decision should approve yes")
	}
	if !strings.Contains(out.String(), "Tool approval required") {
		t.Fatalf("prompt = %q, want approval text", out.String())
	}
}
