package tool

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
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

func TestJSONSchemaForScalarInputWrapsValue(t *testing.T) {
	schema := JSONSchemaFor[string]()

	if schema["type"] != "object" {
		t.Fatalf("type = %v, want object", schema["type"])
	}
	if additional := schema["additionalProperties"]; additional != false {
		t.Fatalf("additionalProperties = %v, want false", additional)
	}

	properties := schema["properties"].(map[string]any)
	value := properties["value"].(map[string]any)
	if value["type"] != "string" {
		t.Fatalf("value.type = %v, want string", value["type"])
	}

	required := schema["required"].([]string)
	if len(required) != 1 || required[0] != "value" {
		t.Fatalf("required = %v, want [value]", required)
	}
}

func TestJSONSchemaForNestedCollectionsAndMaps(t *testing.T) {
	type address struct {
		City string `json:"city"`
	}
	type input struct {
		Addresses []address          `json:"addresses"`
		Scores    map[string]float64 `json:"scores"`
		Tags      [2]string          `json:"tags,omitempty"`
		Any       map[int]string     `json:"any,omitempty"`
	}

	schema := JSONSchemaFor[input]()
	properties := schema["properties"].(map[string]any)

	addresses := properties["addresses"].(map[string]any)
	if addresses["type"] != "array" {
		t.Fatalf("addresses.type = %v, want array", addresses["type"])
	}
	addressItems := addresses["items"].(map[string]any)
	addressProps := addressItems["properties"].(map[string]any)
	city := addressProps["city"].(map[string]any)
	if addressItems["type"] != "object" || city["type"] != "string" {
		t.Fatalf("address items = %#v, want object with string city", addressItems)
	}

	scores := properties["scores"].(map[string]any)
	scoreValues := scores["additionalProperties"].(map[string]any)
	if scores["type"] != "object" || scoreValues["type"] != "number" {
		t.Fatalf("scores schema = %#v, want object with number additionalProperties", scores)
	}

	tags := properties["tags"].(map[string]any)
	tagItems := tags["items"].(map[string]any)
	if tags["type"] != "array" || tagItems["type"] != "string" {
		t.Fatalf("tags schema = %#v, want array of strings", tags)
	}

	any := properties["any"].(map[string]any)
	if _, ok := any["additionalProperties"]; ok {
		t.Fatalf("non-string-key map schema = %#v, want no additionalProperties schema", any)
	}
}

func TestJSONSchemaForTimeAndOmitZero(t *testing.T) {
	type input struct {
		When     time.Time  `json:"when"`
		Optional int        `json:"optional,omitzero"`
		Maybe    *time.Time `json:"maybe"`
	}

	schema := JSONSchemaFor[input]()
	properties := schema["properties"].(map[string]any)

	when := properties["when"].(map[string]any)
	if when["type"] != "string" || when["format"] != "date-time" {
		t.Fatalf("when schema = %#v, want date-time string", when)
	}
	maybe := properties["maybe"].(map[string]any)
	if maybe["type"] != "string" || maybe["format"] != "date-time" {
		t.Fatalf("maybe schema = %#v, want date-time string", maybe)
	}

	required := schema["required"].([]string)
	if len(required) != 1 || required[0] != "when" {
		t.Fatalf("required = %v, want only [when]", required)
	}
}

func TestJSONSchemaForSkippedFieldsAndDefaultNames(t *testing.T) {
	type embedded struct {
		Embedded string `json:"embedded"`
	}
	type input struct {
		embedded
		DefaultName string `json:",omitempty"`
		hidden      string
		Visible     bool
		Skipped     string `json:"-"`
		Ignored     func() `json:"ignored,omitempty"`
	}

	_ = input{hidden: "not exported"}.hidden
	schema := JSONSchemaFor[input]()
	properties := schema["properties"].(map[string]any)

	if _, ok := properties["embedded"]; ok {
		t.Fatal("anonymous embedded field should not appear in schema")
	}
	if _, ok := properties["hidden"]; ok {
		t.Fatal("unexported field should not appear in schema")
	}
	if _, ok := properties["Skipped"]; ok {
		t.Fatal("json:\"-\" field should not appear in schema")
	}
	defaultName := properties["DefaultName"].(map[string]any)
	if defaultName["type"] != "string" {
		t.Fatalf("DefaultName schema = %#v, want string", defaultName)
	}
	visible := properties["Visible"].(map[string]any)
	if visible["type"] != "boolean" {
		t.Fatalf("Visible schema = %#v, want boolean", visible)
	}
	ignored := properties["ignored"].(map[string]any)
	if ignored["type"] != "object" {
		t.Fatalf("unsupported function schema = %#v, want object fallback", ignored)
	}

	required := schema["required"].([]string)
	if len(required) != 1 || required[0] != "Visible" {
		t.Fatalf("required = %v, want only [Visible]", required)
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

func TestToolboxChatToolsAreSortedByName(t *testing.T) {
	toolbox := NewToolbox()
	for _, name := range []string{"zeta", "alpha", "middle"} {
		err := toolbox.RegisterTool(NewTool(name, "Tool "+name+".", func(context.Context, struct{}) (struct{}, error) {
			return struct{}{}, nil
		}))
		if err != nil {
			t.Fatal(err)
		}
	}

	tools := toolbox.ChatTools()
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}

	if strings.Join(names, ",") != "alpha,middle,zeta" {
		t.Fatalf("tool names = %v, want sorted order", names)
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

func TestToolboxPolicyDeniesBeforeHandler(t *testing.T) {
	handlerCalled := false
	toolbox := NewToolbox(WithToolPolicyHook(func(_ context.Context, request ToolPolicyRequest) (ToolPolicyDecision, error) {
		if request.ToolCallID != "call_1" || request.Round != 2 || request.Model != "test-model" {
			t.Fatalf("request = %#v", request)
		}
		return ToolPolicyDecision{Action: ToolPolicyDeny, Reason: "not allowed"}, nil
	}))
	if err := toolbox.RegisterTool(NewTool("dangerous", "Dangerous operation.", func(context.Context, struct{}) (struct{}, error) {
		handlerCalled = true
		return struct{}{}, nil
	})); err != nil {
		t.Fatal(err)
	}

	_, err := toolbox.CallWithRequest(context.Background(), ToolCallRequest{
		ID: "call_1", Name: "dangerous", Arguments: `{}`, Round: 2, Model: "test-model",
	})
	if err == nil || !errors.Is(err, ErrToolPolicyDenied) {
		t.Fatalf("expected policy denial, got %v", err)
	}
	if handlerCalled {
		t.Fatal("handler ran after policy denial")
	}
}

func TestToolboxValidatesArgumentsBeforePolicyAndApproval(t *testing.T) {
	policyCalled := false
	approvalCalled := false
	handlerCalled := false
	toolbox := NewToolbox(
		WithToolPolicyHook(func(context.Context, ToolPolicyRequest) (ToolPolicyDecision, error) {
			policyCalled = true
			return ToolPolicyDecision{Action: ToolPolicyRequireApproval}, nil
		}),
		WithApprovalHook(func(context.Context, ApprovalRequest) (ApprovalDecision, error) {
			approvalCalled = true
			return ApprovalDecision{Approved: true}, nil
		}),
	)
	if err := toolbox.RegisterTool(NewTool("greet", "Greets.", func(_ context.Context, input struct {
		Name string `json:"name"`
	}) (struct{}, error) {
		handlerCalled = true
		return struct{}{}, nil
	})); err != nil {
		t.Fatal(err)
	}

	_, err := toolbox.Call(context.Background(), "greet", `{"name":"Ada","unknown":true}`)
	if err == nil || !strings.Contains(err.Error(), "unknown argument") {
		t.Fatalf("expected validation error, got %v", err)
	}
	if policyCalled || approvalCalled || handlerCalled {
		t.Fatalf("called policy=%v approval=%v handler=%v", policyCalled, approvalCalled, handlerCalled)
	}
}

func TestToolboxRejectsMissingRequiredArgumentsBeforeApproval(t *testing.T) {
	approvalCalled := false
	toolbox := NewToolbox(WithApprovalHook(func(context.Context, ApprovalRequest) (ApprovalDecision, error) {
		approvalCalled = true
		return ApprovalDecision{Approved: true}, nil
	}))
	if err := toolbox.RegisterTool(NewTool("greet", "Greets.", func(_ context.Context, input struct {
		Name string `json:"name"`
	}) (struct{}, error) {
		return struct{}{}, nil
	}, WithApproval(ApprovalPolicy{Mode: ApprovalAlways}))); err != nil {
		t.Fatal(err)
	}

	_, err := toolbox.Call(context.Background(), "greet", `{}`)
	if err == nil || !strings.Contains(err.Error(), "missing required argument name") {
		t.Fatalf("expected required argument error, got %v", err)
	}
	if approvalCalled {
		t.Fatal("approval ran before required argument validation")
	}
}

func TestToolboxStrictestPolicyRequiresApprovalWithCanonicalArguments(t *testing.T) {
	var policyArguments string
	var approvalArguments string
	toolbox := NewToolbox(
		WithToolPolicyHook(func(_ context.Context, request ToolPolicyRequest) (ToolPolicyDecision, error) {
			policyArguments = string(request.Arguments)
			return ToolPolicyDecision{Action: ToolPolicyAllow}, nil
		}),
		WithApprovalHook(func(_ context.Context, request ApprovalRequest) (ApprovalDecision, error) {
			approvalArguments = string(request.Arguments)
			return ApprovalDecision{Approved: true}, nil
		}),
	)
	if err := toolbox.RegisterTool(NewTool("combine", "Combines values.", func(_ context.Context, input struct {
		A int `json:"a"`
		B int `json:"b"`
	}) (int, error) {
		return input.A + input.B, nil
	}, WithApproval(ApprovalPolicy{Mode: ApprovalAlways, Risk: RiskHigh}))); err != nil {
		t.Fatal(err)
	}

	result, err := toolbox.Call(context.Background(), "combine", `{"b":2,"a":1}`)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil || *result != "3" {
		t.Fatalf("result = %v, want 3", result)
	}
	if policyArguments != `{"a":1,"b":2}` || approvalArguments != policyArguments {
		t.Fatalf("policy=%q approval=%q, want canonical arguments", policyArguments, approvalArguments)
	}
}

func TestToolboxRejectsInvalidApprovalPolicyAtRegistration(t *testing.T) {
	tests := []struct {
		name   string
		policy ApprovalPolicy
		want   string
	}{
		{name: "mode", policy: ApprovalPolicy{Mode: ApprovalMode("sometimes")}, want: "invalid approval mode"},
		{name: "risk", policy: ApprovalPolicy{Risk: RiskLevel("critical")}, want: "invalid approval risk"},
		{name: "preview", policy: ApprovalPolicy{Preview: PreviewKind("html")}, want: "invalid approval preview"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			toolbox := NewToolbox()
			err := toolbox.RegisterTool(NewTool("bad", "Bad policy.", func(context.Context, struct{}) (struct{}, error) {
				return struct{}{}, nil
			}, WithApproval(tt.policy)))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected %q, got %v", tt.want, err)
			}
		})
	}
}

func TestToolboxInvalidPolicyDecisionFailsClosed(t *testing.T) {
	toolbox := NewToolbox(WithToolPolicyHook(func(context.Context, ToolPolicyRequest) (ToolPolicyDecision, error) {
		return ToolPolicyDecision{}, nil
	}))
	if err := toolbox.RegisterTool(NewTool("safe", "Safe.", func(context.Context, struct{}) (struct{}, error) {
		return struct{}{}, nil
	})); err != nil {
		t.Fatal(err)
	}
	_, err := toolbox.Call(context.Background(), "safe", `{}`)
	if err == nil || !errors.Is(err, ErrToolPolicyDenied) {
		t.Fatalf("expected fail-closed policy error, got %v", err)
	}
}
