package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

type outputBuffer struct {
	mu     sync.Mutex
	buffer bytes.Buffer
	status chan struct{}
}

func (b *outputBuffer) Write(data []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n, err := b.buffer.Write(data)
	if bytes.Contains(data, []byte("status=running")) {
		select {
		case b.status <- struct{}{}:
		default:
		}
	}
	return n, err
}

func (b *outputBuffer) String() string { b.mu.Lock(); defer b.mu.Unlock(); return b.buffer.String() }

func testConfig(server *httptest.Server) config {
	return config{apiKey: "test-key", model: "test-model", baseURL: server.URL + "/api/v1", task: defaultTask, timeout: 5 * time.Second, statusInterval: time.Hour}
}

func finalSnapshots(t *testing.T, output string) []namedSnapshot {
	t.Helper()
	_, data, ok := strings.Cut(output, "FINAL SNAPSHOTS (independent runs; latest per role)\n")
	if !ok {
		t.Fatalf("missing final snapshots: %s", output)
	}
	var snapshots []namedSnapshot
	if err := json.Unmarshal([]byte(data), &snapshots); err != nil {
		t.Fatal(err)
	}
	return snapshots
}

// Exercise the actual OpenRouter adapter over local SSE; no credentials or paid calls.
func fixtureServer(t *testing.T, gate <-chan struct{}, entered chan<- struct{}, fail bool) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	calls := &atomic.Int64{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model    string `json:"model"`
			Stream   bool   `json:"stream"`
			Provider struct {
				RequireParameters bool `json:"require_parameters"`
			} `json:"provider"`
			Messages []struct {
				Role       string `json:"role"`
				Content    string `json:"content"`
				ToolCallID string `json:"tool_call_id"`
			} `json:"messages"`
			Tools []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
			http.Error(w, "bad body", 400)
			return
		}
		if r.URL.Path != "/api/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer test-key" || request.Model != "test-model" || !request.Stream || !request.Provider.RequireParameters || len(request.Tools) == 0 || len(request.Messages) == 0 {
			t.Errorf("invalid provider request: path=%s model=%s tools=%d", r.URL.Path, request.Model, len(request.Tools))
			http.Error(w, "invalid request", 400)
			return
		}
		if calls.Add(1) == 1 && gate != nil {
			close(entered)
			select {
			case <-gate:
			case <-r.Context().Done():
				return
			}
		}
		if fail {
			http.Error(w, "fixture provider rejected request", 400)
			return
		}
		toolResults := 0
		for _, message := range request.Messages {
			if message.Role == "tool" {
				toolResults++
			}
		}
		var toolName, callID, arguments, content string
		system := request.Messages[0].Content
		switch {
		case strings.HasPrefix(system, "You are the researcher."):
			if toolResults == 0 {
				toolName, callID, arguments = "lookup_catalog", "catalog-1", `{"category":"sensor"}`
			} else {
				if !strings.Contains(request.Messages[len(request.Messages)-1].Content, `"unit_price_cents":2400`) {
					t.Error("catalog result not returned to provider")
				}
				content = "Demo catalog verified: north, south, west."
			}
		case strings.HasPrefix(system, "You are the analyst."):
			if toolResults == 0 {
				toolName, callID, arguments = "calculate_quotes", "quotes-1", `{"quantity":10}`
			} else {
				if !strings.Contains(request.Messages[len(request.Messages)-1].Content, `"total_cents":25000`) {
					t.Error("computed quote not returned to provider")
				}
				content = "Demo west delivered total is 25000 cents for 10 sensors, lowest among available quotes."
			}
		case strings.HasPrefix(system, "You are the coordinator."):
			switch toolResults {
			case 0:
				toolName, callID, arguments = "agent_researcher", "research-1", `{"query":"Verify demo sensor suppliers for 10 units."}`
			case 1:
				if request.Messages[len(request.Messages)-1].ToolCallID != "research-1" || !strings.Contains(request.Messages[len(request.Messages)-1].Content, "Demo catalog verified") {
					t.Error("research delegation result missing")
				}
				toolName, callID, arguments = "agent_analyst", "analysis-1", `{"query":"Demo catalog verified north, south, west. Calculate and compare delivered totals for 10 sensors."}`
			default:
				content = "Demo recommendation: west, 25000 cents delivered for 10 sensors. These are local fixture prices, not live quotes."
			}
		default:
			t.Error("unknown agent system prompt")
			http.Error(w, "unknown agent", 400)
			return
		}
		delta := map[string]any{"role": "assistant", "content": content}
		finish := "stop"
		if toolName != "" {
			if !slices.ContainsFunc(request.Tools, func(tool struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}) bool {
				return tool.Function.Name == toolName
			}) {
				t.Errorf("tool %q not present on request", toolName)
			}
			delta = map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"index": 0, "id": callID, "type": "function", "function": map[string]string{"name": toolName, "arguments": arguments}}}}
			finish = "tool_calls"
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, choice := range []map[string]any{{"index": 0, "delta": delta}, {"index": 0, "delta": map[string]any{}, "finish_reason": finish}} {
			data, _ := json.Marshal(map[string]any{"id": "local-fixture", "object": "chat.completion.chunk", "created": 1, "model": "test-model", "choices": []any{choice}})
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	return server, calls
}

func TestTrackedOpenRouterWorkflow(t *testing.T) {
	gate, entered := make(chan struct{}), make(chan struct{})
	release := sync.OnceFunc(func() { close(gate) })
	defer release()
	server, calls := fixtureServer(t, gate, entered, false)
	defer server.Close()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	out := &outputBuffer{status: make(chan struct{}, 1)}
	done := make(chan error, 1)
	go func() { done <- run(ctx, testConfig(server), out) }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("provider did not start")
	}
	select {
	case <-out.status:
	case <-time.After(5 * time.Second):
		t.Fatal("status blocked behind provider work")
	}
	select {
	case err := <-done:
		t.Fatalf("workflow finished before provider release: %v", err)
	default:
	}
	release()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	output := out.String()
	snapshots := finalSnapshots(t, output)
	if calls.Load() != 7 || len(snapshots) != 3 || !strings.Contains(output, "Demo recommendation: west, 25000 cents") {
		t.Fatalf("calls=%d output=%s", calls.Load(), output)
	}
	ids := make(map[string]bool)
	for _, s := range snapshots {
		if s.Status != "succeeded" || s.RunID == "" || ids[s.RunID] {
			t.Fatalf("run identity/status: %#v", s)
		}
		ids[s.RunID] = true
		switch s.Agent {
		case "coordinator":
			if len(s.CompletedTools) != 2 || len(s.Milestones) != 0 {
				t.Fatal(s)
			}
		case "researcher":
			if !slices.Equal(s.CompletedTools, []string{"lookup_catalog"}) || len(s.Milestones) != 1 || !strings.Contains(s.Milestones[0], "Verified 3") {
				t.Fatal(s)
			}
		case "analyst":
			if !slices.Equal(s.CompletedTools, []string{"calculate_quotes"}) || len(s.Milestones) != 1 || !strings.Contains(s.Milestones[0], "3 available demo quotes") {
				t.Fatal(s)
			}
		}
	}
	if strings.Contains(output, "test-key") {
		t.Fatal("credential in output")
	}
	var trace []string
	for _, line := range strings.Split(output, "\n") {
		if strings.HasPrefix(line, "STATUS") || strings.HasPrefix(line, "Demo recommendation:") {
			trace = append(trace, line)
		}
	}
	t.Logf("LOCAL HTTP SSE FIXTURE through the real OpenRouter adapter; no live/paid calls. Requests: %d\n%s", calls.Load(), strings.Join(trace, "\n"))
	for _, s := range snapshots {
		t.Logf("FINAL %s status=%s completed=%v milestones=%v", s.Agent, s.Status, s.CompletedTools, s.Milestones)
	}
}

func TestTrackedOpenRouterFailureAndCancellation(t *testing.T) {
	t.Run("provider_failure", func(t *testing.T) {
		server, _ := fixtureServer(t, nil, nil, true)
		defer server.Close()
		var out bytes.Buffer
		if err := run(t.Context(), testConfig(server), &out); err == nil {
			t.Fatal("expected provider failure")
		}
		s := finalSnapshots(t, out.String())
		if len(s) != 1 || s[0].Agent != "coordinator" || s[0].Status != "failed" || strings.Contains(out.String(), "FINAL ANSWER") {
			t.Fatal(out.String())
		}
	})
	t.Run("canceled", func(t *testing.T) {
		gate, entered := make(chan struct{}), make(chan struct{})
		server, calls := fixtureServer(t, gate, entered, false)
		defer server.Close()
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		out := &outputBuffer{status: make(chan struct{}, 1)}
		done := make(chan error, 1)
		go func() { done <- run(ctx, testConfig(server), out) }()
		select {
		case <-entered:
		case <-time.After(5 * time.Second):
			t.Fatal("provider did not start")
		}
		cancel()
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v", err)
		}
		s := finalSnapshots(t, out.String())
		if len(s) != 1 || s[0].Status != "failed" || calls.Load() != 1 {
			t.Fatalf("snapshots=%#v calls=%d", s, calls.Load())
		}
	})
}

func TestRunTrackingReplacesOnlyTheRepeatedRole(t *testing.T) {
	tracking := newRunTracking()
	coordinator, researcher := tracking.observer("coordinator"), tracking.observer("researcher")
	emit := func(observer llm.ProgressObserver, id string, kind llm.ProgressKind) {
		observer.OnProgress(llm.ProgressEvent{RunID: id, Kind: kind, Time: time.Now().UTC()})
	}
	emit(coordinator, "parent", llm.ProgressRunStarted)
	emit(researcher, "first", llm.ProgressRunStarted)
	emit(researcher, "first", llm.ProgressRunSucceeded)
	for i := 0; i < 200; i++ {
		id := fmt.Sprint(i)
		emit(researcher, id, llm.ProgressRunStarted)
		emit(researcher, id, llm.ProgressRunSucceeded)
	}
	emit(researcher, "first", llm.ProgressRunFailed)
	s := tracking.snapshots()
	if len(tracking.latest) != 2 || s[0].RunID != "parent" || s[0].Status != "running" || s[1].RunID != "199" || s[1].Status != "succeeded" {
		t.Fatal(s)
	}
}

func TestExampleConfigurationAndDemoTools(t *testing.T) {
	for _, cfg := range []config{{}, {apiKey: "key", model: "model", task: "", timeout: time.Second, statusInterval: time.Second}, {apiKey: "key", model: "model", task: "task", timeout: -time.Second, statusInterval: time.Second}, {apiKey: "key", model: "model", task: "task", timeout: time.Second, statusInterval: 0}} {
		if err := run(t.Context(), cfg, &bytes.Buffer{}); err == nil {
			t.Fatal("accepted invalid configuration")
		}
	}
	for _, q := range []int{0, 101} {
		if _, err := calculateQuotes(t.Context(), quoteInput{q}); err == nil {
			t.Fatal("accepted invalid quantity")
		}
	}
	quotes, err := calculateQuotes(t.Context(), quoteInput{10})
	if err != nil || len(quotes) != 3 || quotes[2].SupplierID != "west" || quotes[2].TotalCents != 25000 {
		t.Fatalf("quotes=%v err=%v", quotes, err)
	}
	if _, err := lookupCatalog(t.Context(), catalogInput{"unknown"}); err == nil {
		t.Fatal("accepted unknown category")
	}
}
