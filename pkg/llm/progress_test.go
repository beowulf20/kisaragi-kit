package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

type progressRecorder struct {
	mu     sync.Mutex
	events []ProgressEvent
}

func (r *progressRecorder) OnProgress(event ProgressEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, event)
}

func (r *progressRecorder) snapshot() []ProgressEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.events)
}

func progressFixture(t *testing.T) (CompletionCallInput, *fakeChatClient, *llmtool.Tool) {
	t.Helper()
	tool := llmtool.NewTool("work", "offline work", func(ctx context.Context, _ struct {
		Value string `json:"value"`
	}) (string, error) {
		ReportMilestone(ctx, "Verified 3 records")
		return "RESULT_SECRET", nil
	})
	client := &fakeChatClient{responses: []*ChatResponse{
		{Reasoning: "REASON_SECRET", ToolCalls: []ToolCall{{ID: "call-1", Name: "work", Arguments: `{"value":"ARG_SECRET"}`}}},
		{Content: "ANSWER_SECRET"},
	}}
	input := CompletionCallInput{Client: client, Model: "fake", Messages: []Message{NewSystemMessage("PROMPT_SECRET"), NewUserMessage("work")}}
	return input, client, &tool
}

func registerProgressTool(t *testing.T, input *CompletionCallInput, tool llmtool.Tool, options ...llmtool.ToolboxOption) {
	t.Helper()
	box := llmtool.NewToolbox(options...)
	if err := box.RegisterTool(tool); err != nil {
		t.Fatal(err)
	}
	input.Tools = *box
}

func checkProgressRun(t *testing.T, events []ProgressEvent, terminal ProgressKind) {
	t.Helper()
	if len(events) < 2 || events[0].Kind != ProgressRunStarted || events[len(events)-1].Kind != terminal {
		t.Fatalf("events = %#v", events)
	}
	var generation, tool, terminals int
	for i, event := range events {
		if event.RunID == "" || event.RunID != events[0].RunID || event.Time.Location() != time.UTC || (i > 0 && event.Time.Before(events[i-1].Time)) {
			t.Fatalf("invalid identity/time: %#v", event)
		}
		switch event.Kind {
		case ProgressGenerationStarted:
			generation++
		case ProgressGenerationFinished:
			generation--
		case ProgressToolStarted:
			tool++
		case ProgressToolSucceeded, ProgressToolFailed:
			tool--
		case ProgressRunSucceeded, ProgressRunFailed:
			terminals++
		}
		if generation < 0 || generation > 1 || tool < 0 || tool > 1 {
			t.Fatalf("unpaired event: %#v", event)
		}
		if event.Kind != ProgressMilestone && event.Message != "" {
			t.Fatalf("unexpected message: %#v", event)
		}
	}
	if generation != 0 || tool != 0 || terminals != 1 {
		t.Fatalf("unbalanced events: %#v", events)
	}
}

func TestProgressObserverSuccessFlowAndSafePayload(t *testing.T) {
	input, client, tool := progressFixture(t)
	registerProgressTool(t, &input, *tool)
	client.onComplete = func(_ ChatRequest, hooks CompletionHooks) {
		_ = hooks.EmitContentDeltaEvent("STREAM_SECRET")
		_ = hooks.EmitReasoningDelta("STREAM_REASON_SECRET")
	}
	r := &progressRecorder{}
	input.ProgressObserver = r
	out, err := Completion(input)
	if err != nil || out.Content != "ANSWER_SECRET" {
		t.Fatalf("output=%#v err=%v", out, err)
	}
	events := r.snapshot()
	checkProgressRun(t, events, ProgressRunSucceeded)
	want := []ProgressKind{ProgressRunStarted, ProgressGenerationStarted, ProgressGenerationFinished, ProgressToolStarted, ProgressMilestone, ProgressToolSucceeded, ProgressGenerationStarted, ProgressGenerationFinished, ProgressRunSucceeded}
	var kinds []ProgressKind
	for _, event := range events {
		kinds = append(kinds, event.Kind)
	}
	if !slices.Equal(kinds, want) {
		t.Fatalf("kinds=%v", kinds)
	}
	if events[4].ToolCallID != "call-1" || events[4].ToolName != "work" || events[4].Message != "Verified 3 records" || events[6].Round != 1 {
		t.Fatalf("metadata=%#v", events)
	}
	data, _ := json.Marshal(events)
	for _, secret := range []string{"PROMPT_SECRET", "ARG_SECRET", "RESULT_SECRET", "REASON_SECRET", "ANSWER_SECRET", "STREAM_SECRET", "STREAM_REASON_SECRET"} {
		if strings.Contains(string(data), secret) {
			t.Fatalf("leaked %s", secret)
		}
	}
}

func TestProgressObserverToolOutcomes(t *testing.T) {
	for _, tc := range []struct {
		name       string
		outcome    ProgressKind
		category   string
		runFailure bool
	}{
		{"error", ProgressToolFailed, "tool_error", false},
		{"nil_result", ProgressToolFailed, "invalid_result", false},
		{"handler_panic", ProgressToolFailed, "tool_error", false},
		{"invalid_args", ProgressToolFailed, "tool_error", false},
		{"unknown_tool", ProgressToolFailed, "tool_error", false},
		{"policy", ProgressToolFailed, "policy_denied", true},
		{"approval", ProgressToolFailed, "approval_denied", false},
		{"approval_limit", ProgressToolFailed, "limit_exceeded", true},
		{"interceptor", ProgressToolFailed, "tool_error", true},
		{"result_hook", ProgressToolFailed, "hook_aborted", true},
		{"error_hook", ProgressToolFailed, "hook_aborted", true},
		{"result_guard", ProgressToolFailed, "guardrail_blocked", true},
		{"approval_guard", ProgressToolFailed, "guardrail_blocked", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input, client, tool := progressFixture(t)
			var options []llmtool.ToolboxOption
			failure := errors.New("ERROR_SECRET")
			switch tc.name {
			case "error", "interceptor", "error_hook":
				tool.Call = func(context.Context, json.RawMessage) (*string, error) { return nil, failure }
			case "nil_result":
				tool.Call = func(context.Context, json.RawMessage) (*string, error) { return nil, nil }
			case "handler_panic":
				tool.Call = func(context.Context, json.RawMessage) (*string, error) { panic("PANIC_SECRET") }
			case "invalid_args":
				client.responses[0].ToolCalls[0].Arguments = "{"
			case "unknown_tool":
				client.responses[0].ToolCalls[0].Name = "missing"
			case "policy":
				options = append(options, llmtool.WithToolPolicyHook(func(context.Context, llmtool.ToolPolicyRequest) (llmtool.ToolPolicyDecision, error) {
					return llmtool.ToolPolicyDecision{Action: llmtool.ToolPolicyDeny}, nil
				}))
			}
			if strings.HasPrefix(tc.name, "approval") {
				tool.Approval.Mode = llmtool.ApprovalAlways
			}
			if tc.name == "approval_limit" {
				input.MaxApprovalDenials = 1
			}
			if tc.name == "interceptor" {
				input.ToolErrorInterceptor = func(ToolErrorContext) ToolErrorDecision { return ToolErrorDecision{Abort: true} }
			}
			if tc.name == "result_hook" || tc.name == "error_hook" {
				input.Hooks.OnEvent = func(e Event) error {
					if _, ok := e.(EventToolResult); ok && tc.name == "result_hook" {
						return failure
					}
					if _, ok := e.(EventToolError); ok && tc.name == "error_hook" {
						return failure
					}
					return nil
				}
			}
			if tc.name == "result_guard" || tc.name == "approval_guard" {
				phase := MessageGuardrailPhaseToolResult
				if tc.name == "approval_guard" {
					phase = MessageGuardrailPhaseApprovalDecision
					input.ApprovalDecisionMessages.AppendAccepted = true
					options = append(options, llmtool.WithApprovalHook(func(context.Context, llmtool.ApprovalRequest) (llmtool.ApprovalDecision, error) {
						return llmtool.ApprovalDecision{Approved: true}, nil
					}))
				}
				input.MessageGuardrails = []MessageGuardrail{NewMessageGuardrail("block", func(_ context.Context, in MessageGuardrailInput) (MessageGuardrailDecision, error) {
					if in.Phase == phase {
						return MessageGuardrailDecision{Action: MessageGuardrailBlock}, nil
					}
					return MessageGuardrailDecision{Action: MessageGuardrailAllow}, nil
				})}
			}
			registerProgressTool(t, &input, *tool, options...)
			r := &progressRecorder{}
			input.ProgressObserver = r
			out, err := Completion(input)
			terminal := ProgressRunSucceeded
			if tc.runFailure {
				terminal = ProgressRunFailed
				if err == nil {
					t.Fatal("expected failure")
				}
			} else if err != nil || out.Content != "ANSWER_SECRET" {
				t.Fatalf("output=%#v err=%v", out, err)
			}
			events := r.snapshot()
			checkProgressRun(t, events, terminal)
			var outcomes []ProgressEvent
			for _, event := range events {
				if event.Kind == ProgressToolSucceeded || event.Kind == ProgressToolFailed {
					outcomes = append(outcomes, event)
				}
			}
			if len(outcomes) != 1 || outcomes[0].Kind != tc.outcome || outcomes[0].ErrorKind != tc.category {
				t.Fatalf("outcomes=%#v", outcomes)
			}
			if tc.runFailure && events[len(events)-1].ErrorKind != tc.category {
				t.Fatalf("terminal=%#v", events[len(events)-1])
			}
			data, _ := json.Marshal(events)
			if strings.Contains(string(data), "ERROR_SECRET") || strings.Contains(string(data), "PANIC_SECRET") {
				t.Fatal("raw error leaked")
			}
		})
	}
}

func TestProgressObserverProviderFailures(t *testing.T) {
	for _, nilResponse := range []bool{false, true} {
		t.Run(fmt.Sprint(nilResponse), func(t *testing.T) {
			input, client, _ := progressFixture(t)
			wantAttempts, category := 3, "provider_error"
			if nilResponse {
				client.responses = []*ChatResponse{nil}
				wantAttempts, category = 1, "invalid_result"
			} else {
				client.errors = []error{errors.New("SECRET"), errors.New("SECRET"), errors.New("SECRET")}
			}
			r := &progressRecorder{}
			input.ProgressObserver = r
			if _, err := Completion(input); err == nil {
				t.Fatal("expected error")
			}
			events := r.snapshot()
			checkProgressRun(t, events, ProgressRunFailed)
			if len(client.requests) != wantAttempts || events[len(events)-1].ErrorKind != category {
				t.Fatalf("events=%#v attempts=%d", events, len(client.requests))
			}
			for i := 0; i < wantAttempts; i++ {
				if events[1+2*i].Attempt != i || events[2+2*i].ErrorKind != category {
					t.Fatalf("attempt events=%#v", events)
				}
			}
		})
	}
}

func TestProgressObserverPanicIsolationAndNilCompatibility(t *testing.T) {
	for _, mode := range []string{"success", "tool_error", "abort"} {
		t.Run(mode, func(t *testing.T) {
			run := func(observer ProgressObserver) (*CompletionCallOutput, []string, []ChatRequest, error) {
				input, client, tool := progressFixture(t)
				if mode == "tool_error" {
					tool.Call = func(context.Context, json.RawMessage) (*string, error) { return nil, errors.New("failure") }
				}
				var hooks []string
				input.Hooks.OnEvent = func(event Event) error {
					hooks = append(hooks, fmt.Sprintf("%T", event))
					if mode == "abort" {
						if _, ok := event.(EventToolResult); ok {
							return errors.New("abort")
						}
					}
					return nil
				}
				input.ProgressObserver = observer
				registerProgressTool(t, &input, *tool)
				out, err := Completion(input)
				return out, hooks, client.requests, err
			}
			base, baseHooks, baseRequests, baseErr := run(nil)
			r := &progressRecorder{}
			out, hooks, requests, err := run(ProgressObserverFunc(func(event ProgressEvent) { r.OnProgress(event); panic("observer failed") }))
			if !reflect.DeepEqual(base, out) || fmt.Sprint(baseErr) != fmt.Sprint(err) || !slices.Equal(baseHooks, hooks) || !reflect.DeepEqual(baseRequests, requests) {
				t.Fatalf("observation changed behavior: base=%#v/%v observed=%#v/%v", base, baseErr, out, err)
			}
			terminal := ProgressRunSucceeded
			if err != nil {
				terminal = ProgressRunFailed
			}
			checkProgressRun(t, r.snapshot(), terminal)
		})
	}
}

func TestProgressObserverCanceled(t *testing.T) {
	for _, observed := range []bool{false, true} {
		for _, when := range []string{"before", "during", "before_provider", "between_retries", "deadline"} {
			t.Run(fmt.Sprintf("%t/%s", observed, when), func(t *testing.T) {
				input, client, _ := progressFixture(t)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				input.Context = ctx
				wantAttempts, category := 1, "canceled"
				wrapped := fmt.Errorf("provider stopped: %w", context.Canceled)
				client.errors = []error{wrapped}
				switch when {
				case "before":
					cancel()
					wantAttempts = 0
				case "during":
					client.onComplete = func(ChatRequest, CompletionHooks) { cancel() }
				case "before_provider":
					input.Hooks.OnGenerationStart = func(GenerationStartEvent) { cancel() }
					wantAttempts = 0
				case "between_retries":
					client.errors = []error{errors.New("transient")}
					input.Hooks.OnCallError = func(error) { cancel() }
				case "deadline":
					expired, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
					defer stop()
					input.Context = expired
					wantAttempts, category = 0, "deadline_exceeded"
				}
				r := &progressRecorder{}
				if observed {
					input.ProgressObserver = r
				}
				_, err := Completion(input)
				wantErr := error(context.Canceled)
				if category == "deadline_exceeded" {
					wantErr = context.DeadlineExceeded
				}
				if !errors.Is(err, wantErr) || len(client.requests) != wantAttempts {
					t.Fatalf("attempts=%d err=%v", len(client.requests), err)
				}
				if when == "during" && !errors.Is(err, wrapped) {
					t.Fatal("lost provider cancellation wrapper")
				}
				if observed {
					events := r.snapshot()
					checkProgressRun(t, events, ProgressRunFailed)
					if events[len(events)-1].ErrorKind != category {
						t.Fatalf("events=%#v", events)
					}
				}
			})
		}
	}
	for _, observed := range []bool{false, true} {
		t.Run(fmt.Sprintf("provider_local_timeout/%t", observed), func(t *testing.T) {
			input, client, _ := progressFixture(t)
			client.responses = []*ChatResponse{{Content: "recovered"}}
			client.errors = []error{context.DeadlineExceeded, context.DeadlineExceeded}
			if observed {
				input.ProgressObserver = &progressRecorder{}
			}
			out, err := Completion(input)
			if err != nil || out.Content != "recovered" || len(client.requests) != 3 {
				t.Fatalf("output=%#v attempts=%d err=%v", out, len(client.requests), err)
			}
		})
	}
}

func TestProgressObserverEarlyExits(t *testing.T) {
	for _, where := range []string{"validation", "nil_client", "input_guard", "generation_start_hook", "tool_call_hook", "tool_limit", "provider_limit", "token_usage", "round_limit"} {
		t.Run(where, func(t *testing.T) {
			input, client, tool := progressFixture(t)
			registerProgressTool(t, &input, *tool)
			r := &progressRecorder{}
			input.ProgressObserver = r
			category := "hook_aborted"
			switch where {
			case "validation":
				input.Model = ""
			case "nil_client":
				input.Client = nil
			case "input_guard":
				category = "guardrail_blocked"
				input.MessageGuardrails = []MessageGuardrail{NewMessageGuardrail("block", func(context.Context, MessageGuardrailInput) (MessageGuardrailDecision, error) {
					return MessageGuardrailDecision{Action: MessageGuardrailBlock}, nil
				})}
			case "generation_start_hook":
				input.Hooks.OnEvent = func(e Event) error {
					if _, ok := e.(EventGenerationStart); ok {
						return errors.New("abort")
					}
					return nil
				}
			case "tool_call_hook":
				input.Hooks.OnEvent = func(e Event) error {
					if _, ok := e.(EventToolCall); ok {
						return errors.New("abort")
					}
					return nil
				}
			case "tool_limit":
				category = "limit_exceeded"
				input.MaxToolCalls = 1
				client.responses[0].ToolCalls = append(client.responses[0].ToolCalls, ToolCall{ID: "second", Name: "work", Arguments: `{"value":"two"}`})
			case "provider_limit":
				category = "limit_exceeded"
				input.MaxProviderAttempts = 1
				client.errors = []error{errors.New("retry")}
			case "token_usage":
				category = "usage_unavailable"
				input.MaxTotalTokens = 1
			case "round_limit":
				category = "limit_exceeded"
				input.MaxToolCallRounds = 1
				client.responses[1] = client.responses[0]
			}
			if _, err := Completion(input); err == nil {
				t.Fatal("expected failure")
			}
			events := r.snapshot()
			if where == "validation" || where == "nil_client" {
				if len(events) != 0 {
					t.Fatal(events)
				}
				return
			}
			checkProgressRun(t, events, ProgressRunFailed)
			if events[len(events)-1].ErrorKind != category {
				t.Fatalf("events=%#v", events)
			}
			if where == "tool_call_hook" {
				for _, event := range events {
					if event.Kind == ProgressToolStarted {
						t.Fatal("tool started before hook accepted")
					}
				}
			}
		})
	}
}

func TestProgressObserverPreservesExecutionPanics(t *testing.T) {
	for _, where := range []string{"provider", "generation_start", "generation_end", "result_hook", "policy", "interceptor"} {
		t.Run(where, func(t *testing.T) {
			input, client, tool := progressFixture(t)
			original := &struct{ source string }{where}
			var options []llmtool.ToolboxOption
			switch where {
			case "provider":
				client.onComplete = func(ChatRequest, CompletionHooks) { panic(original) }
			case "generation_start":
				input.Hooks.OnGenerationStart = func(GenerationStartEvent) { panic(original) }
			case "generation_end":
				input.Hooks.OnGenerationEnd = func(GenerationEndEvent) { panic(original) }
			case "result_hook":
				input.Hooks.OnToolResult = func(ToolCall) { panic(original) }
			case "policy":
				options = append(options, llmtool.WithToolPolicyHook(func(context.Context, llmtool.ToolPolicyRequest) (llmtool.ToolPolicyDecision, error) { panic(original) }))
			case "interceptor":
				tool.Call = func(context.Context, json.RawMessage) (*string, error) { return nil, errors.New("tool failure") }
				input.ToolErrorInterceptor = func(ToolErrorContext) ToolErrorDecision { panic(original) }
			}
			registerProgressTool(t, &input, *tool, options...)
			r := &progressRecorder{}
			input.ProgressObserver = r
			func() {
				defer func() {
					if got := recover(); got != original {
						t.Fatalf("panic=%#v want %#v", got, original)
					}
				}()
				_, _ = Completion(input)
			}()
			events := r.snapshot()
			checkProgressRun(t, events, ProgressRunFailed)
			if events[len(events)-1].ErrorKind != "panic" {
				t.Fatal(events)
			}
		})
	}
}

func TestReportMilestoneScopeAndConcurrency(t *testing.T) {
	ReportMilestone(context.Background(), "outside")
	input, _, tool := progressFixture(t)
	var saved context.Context
	tool.Call = func(ctx context.Context, _ json.RawMessage) (*string, error) {
		saved = ctx
		ReportMilestone(ctx, " ")
		var wg sync.WaitGroup
		for i := 0; i < 32; i++ {
			wg.Add(1)
			go func(i int) { defer wg.Done(); ReportMilestone(ctx, fmt.Sprint("verified ", i)) }(i)
		}
		wg.Wait()
		result := "done"
		return &result, nil
	}
	registerProgressTool(t, &input, *tool)
	r := &progressRecorder{}
	input.ProgressObserver = r
	input.Hooks.OnToolResult = func(ToolCall) { ReportMilestone(saved, "after handler returned") }
	if _, err := Completion(input); err != nil {
		t.Fatal(err)
	}
	before := r.snapshot()
	ReportMilestone(saved, "after run returned")
	if len(r.snapshot()) != len(before) {
		t.Fatal("late milestone accepted")
	}
	count := 0
	for _, event := range before {
		if event.Kind == ProgressMilestone {
			count++
			if event.ToolCallID != "call-1" || event.Round != 0 || event.Attempt != 0 {
				t.Fatal(event)
			}
		}
	}
	if count != 32 {
		t.Fatalf("milestones=%d", count)
	}
	checkProgressRun(t, before, ProgressRunSucceeded)
}
