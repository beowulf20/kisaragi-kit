package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/progress"
)

type outputBuffer struct {
	mu        sync.Mutex
	buffer    bytes.Buffer
	status    chan struct{}
	heartbeat chan struct{}
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
	if bytes.Contains(data, []byte("HEARTBEAT")) {
		select {
		case b.heartbeat <- struct{}{}:
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
			switch toolResults {
			case 0:
				toolName, callID, arguments = "lookup_catalog", "catalog-1", `{"category":"sensor"}`
			case 1:
				if !strings.Contains(request.Messages[len(request.Messages)-1].Content, `"unit_price_cents":2400`) {
					t.Error("catalog result not returned to provider")
				}
				toolName, callID, arguments = "check_inventory", "inventory-1", `{"quantity":10}`
			case 2:
				if !strings.Contains(request.Messages[len(request.Messages)-1].Content, `"sufficient":true`) {
					t.Error("inventory result missing")
				}
				toolName, callID, arguments = "check_delivery", "delivery-1", `{"max_days":7}`
			default:
				if !strings.Contains(request.Messages[len(request.Messages)-1].Content, `"within_deadline":false`) {
					t.Error("delivery exclusion missing")
				}
				content = "Demo catalog verified: north, south, west. All can stock 10, but south misses the 7-day deadline."
			}
		case strings.HasPrefix(system, "You are the analyst."):
			switch toolResults {
			case 0:
				toolName, callID, arguments = "calculate_quotes", "quotes-1", `{"quantity":10}`
			case 1:
				if !strings.Contains(request.Messages[len(request.Messages)-1].Content, `"total_cents":25000`) {
					t.Error("computed quote not returned to provider")
				}
				toolName, callID, arguments = "evaluate_scenarios", "scenarios-1", `{"quantities":[10,20,30],"max_days":7}`
			case 2:
				if !strings.Contains(request.Messages[len(request.Messages)-1].Content, `"total_cents":73200`) {
					t.Error("30-unit scenario result missing")
				}
				toolName, callID, arguments = "audit_recommendation", "audit-1", `{"supplier_id":"west","quantity":10,"max_days":7,"total_cents":25000}`
			default:
				if !strings.Contains(request.Messages[len(request.Messages)-1].Content, `"approved":true`) {
					t.Error("recommendation audit missing")
				}
				content = "Audited demo west total: 25000 cents for 10 sensors. Scenarios: west for 10; north for 20/30 under the deadline."
			}
		case strings.HasPrefix(system, "You are the coordinator."):
			switch toolResults {
			case 0:
				toolName, callID, arguments = "agent_researcher", "research-1", `{"query":"Verify demo catalog, stock for 10 sensors, and delivery under a 7-day deadline."}`
			case 1:
				if request.Messages[len(request.Messages)-1].ToolCallID != "research-1" || !strings.Contains(request.Messages[len(request.Messages)-1].Content, "Demo catalog verified") {
					t.Error("research delegation result missing")
				}
				toolName, callID, arguments = "agent_analyst", "analysis-1", `{"query":"Demo catalog verified north/south/west; south misses 7 days. Quote primary quantity 10, compare scenarios 10/20/30 under 7 days, and audit the primary recommendation."}`
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
	if calls.Load() != 11 || len(snapshots) != 3 || !strings.Contains(output, "Demo recommendation: west, 25000 cents") {
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
			if !slices.Equal(s.CompletedTools, []string{"lookup_catalog", "check_inventory", "check_delivery"}) || len(s.Milestones) != 7 || !strings.Contains(s.Milestones[0], "Verified 3") {
				t.Fatal(s)
			}
		case "analyst":
			if !slices.Equal(s.CompletedTools, []string{"calculate_quotes", "evaluate_scenarios", "audit_recommendation"}) || len(s.Milestones) != 9 || !strings.Contains(s.Milestones[8], "approved=true") {
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
	demo := demoTools{}
	for _, cfg := range []config{{}, {apiKey: "key", model: "model", task: "", timeout: time.Second, statusInterval: time.Second}, {apiKey: "key", model: "model", task: "task", timeout: -time.Second, statusInterval: time.Second}, {apiKey: "key", model: "model", task: "task", timeout: time.Second, statusInterval: 0}} {
		if err := run(t.Context(), cfg, &bytes.Buffer{}); err == nil {
			t.Fatal("accepted invalid configuration")
		}
	}
	for _, q := range []int{0, 101} {
		if _, err := demo.calculateQuotes(t.Context(), quoteInput{q}); err == nil {
			t.Fatal("accepted invalid quantity")
		}
	}
	quotes, err := demo.calculateQuotes(t.Context(), quoteInput{10})
	if err != nil || len(quotes) != 3 || quotes[2].SupplierID != "west" || quotes[2].TotalCents != 25000 {
		t.Fatalf("quotes=%v err=%v", quotes, err)
	}
	if _, err := demo.lookupCatalog(t.Context(), catalogInput{"unknown"}); err == nil {
		t.Fatal("accepted unknown category")
	}
}

func TestHeartbeatLeavesTrackerUnchanged(t *testing.T) {
	tracking := newRunTracking()
	observer := tracking.observer("coordinator")
	observer.OnProgress(llm.ProgressEvent{RunID: "one", Kind: llm.ProgressRunStarted, Time: time.Now().UTC()})
	previous := make(map[string]progress.Snapshot)
	var out bytes.Buffer
	if err := tracking.printChanges(&out, previous, false); err != nil {
		t.Fatal(err)
	}
	before := tracking.snapshots()[0].Snapshot
	out.Reset()
	if err := tracking.printChanges(&out, previous, false); err != nil || out.Len() != 0 {
		t.Fatal("unchanged notification emitted output")
	}
	for i := 0; i < 2; i++ {
		if err := tracking.printChanges(&out, previous, true); err != nil {
			t.Fatal(err)
		}
	}
	if strings.Count(out.String(), "HEARTBEAT") != 2 || !strings.Contains(out.String(), "still_running") || !strings.Contains(out.String(), "version=1") {
		t.Fatal(out.String())
	}
	if !reflect.DeepEqual(before, tracking.snapshots()[0].Snapshot) {
		t.Fatal("heartbeat changed progress state")
	}
	observer.OnProgress(llm.ProgressEvent{RunID: "one", Kind: llm.ProgressRunSucceeded, Time: time.Now().UTC()})
	if err := tracking.printChanges(&out, previous, false); err != nil {
		t.Fatal(err)
	}
	frozen := tracking.snapshots()[0].Elapsed
	out.Reset()
	if err := tracking.printChanges(&out, previous, true); err != nil || out.Len() != 0 || tracking.snapshots()[0].Elapsed != frozen {
		t.Fatal("terminal run heartbeat/elapsed changed")
	}
}

func TestPeriodicHeartbeatDuringProviderWait(t *testing.T) {
	gate, entered := make(chan struct{}), make(chan struct{})
	server, _ := fixtureServer(t, gate, entered, false)
	defer server.Close()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	cfg := testConfig(server)
	cfg.statusInterval = 2 * time.Millisecond
	out := &outputBuffer{status: make(chan struct{}, 1), heartbeat: make(chan struct{}, 1)}
	done := make(chan error, 1)
	go func() { done <- run(ctx, cfg, out) }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("provider did not start")
	}
	select {
	case <-out.heartbeat:
	case <-time.After(5 * time.Second):
		t.Fatal("no heartbeat while provider waited")
	}
	var lastVersion string
	found := false
	for _, line := range strings.Split(out.String(), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		if fields[0] == "STATUS" {
			lastVersion = fields[len(fields)-1]
		}
		if fields[0] == "HEARTBEAT" {
			if fields[len(fields)-1] != lastVersion {
				t.Fatalf("heartbeat changed version: %s", line)
			}
			found = true
			break
		}
	}
	if !found {
		t.Fatal("heartbeat output missing")
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	s := finalSnapshots(t, out.String())
	if len(s) != 1 || len(s[0].Milestones) != 0 || s[0].Status != "failed" {
		t.Fatal(s)
	}
}

func TestPacingCancellationAndFastMode(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	start := time.Now()
	if err := (demoTools{delay: time.Hour}).step(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error=%v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("pacing ignored cancellation")
	}
	if err := (demoTools{}).step(t.Context()); err != nil {
		t.Fatal(err)
	}
	server, _ := fixtureServer(t, nil, nil, false)
	defer server.Close()
	cfg := testConfig(server)
	cfg.stepDelay = -time.Second
	if err := run(t.Context(), cfg, &bytes.Buffer{}); err == nil {
		t.Fatal("accepted negative step delay")
	}
}

func TestScenarioAndRecommendationFacts(t *testing.T) {
	demo := demoTools{}
	stock, err := demo.checkInventory(t.Context(), quoteInput{30})
	if err != nil || len(stock) != 3 || stock[2].Sufficient || !stock[0].Sufficient {
		t.Fatalf("stock=%v err=%v", stock, err)
	}
	delivery, err := demo.checkDelivery(t.Context(), deliveryInput{7})
	if err != nil || len(delivery) != 3 || delivery[1].WithinDeadline || !delivery[2].WithinDeadline {
		t.Fatalf("delivery=%v err=%v", delivery, err)
	}
	scenarios, err := demo.evaluateScenarios(t.Context(), scenarioInput{Quantities: []int{10, 20, 30, 50}, MaxDays: 7})
	if err != nil || len(scenarios) != 4 {
		t.Fatalf("scenarios=%v err=%v", scenarios, err)
	}
	for i, want := range []struct {
		id    string
		total int
	}{{"west", 25000}, {"north", 49200}, {"north", 73200}} {
		if scenarios[i].Cheapest == nil || scenarios[i].Cheapest.SupplierID != want.id || scenarios[i].Cheapest.TotalCents != want.total {
			t.Fatal(scenarios)
		}
	}
	if scenarios[3].Cheapest != nil {
		t.Fatal("invented an unavailable supplier")
	}
	for _, tc := range []struct {
		name     string
		input    auditInput
		approved bool
	}{
		{"correct", auditInput{"west", 10, 7, 25000}, true},
		{"tied_cheapest", auditInput{"west", 12, 7, 30000}, true},
		{"wrong_total", auditInput{"west", 10, 7, 24999}, false},
		{"no_stock", auditInput{"west", 30, 7, 75000}, false},
		{"late", auditInput{"south", 10, 7, 25400}, false},
		{"not_cheapest", auditInput{"north", 10, 7, 25200}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := demo.auditRecommendation(t.Context(), tc.input)
			if err != nil || result.Approved != tc.approved {
				t.Fatalf("audit=%#v err=%v", result, err)
			}
			if result.Approved && result.CheapestSupplierID != tc.input.SupplierID {
				t.Fatal("approved supplier not identified")
			}
		})
	}
	for _, input := range []scenarioInput{{Quantities: nil, MaxDays: 7}, {Quantities: []int{10, 10}, MaxDays: 7}, {Quantities: []int{0}, MaxDays: 7}, {Quantities: []int{101}, MaxDays: 7}, {Quantities: []int{1, 2, 3, 4, 5, 6}, MaxDays: 7}, {Quantities: []int{10}, MaxDays: 0}} {
		if _, err := demo.evaluateScenarios(t.Context(), input); err == nil {
			t.Fatal("accepted invalid scenarios")
		}
	}
	if _, err := demo.checkInventory(t.Context(), quoteInput{0}); err == nil {
		t.Fatal("accepted invalid stock quantity")
	}
	if _, err := demo.checkDelivery(t.Context(), deliveryInput{61}); err == nil {
		t.Fatal("accepted invalid delivery deadline")
	}
	for _, input := range []auditInput{{"unknown", 10, 7, 25000}, {"west", 0, 7, 25000}, {"west", 10, 0, 25000}, {"west", 10, 7, -1}} {
		if _, err := demo.auditRecommendation(t.Context(), input); err == nil {
			t.Fatal("accepted invalid audit")
		}
	}
}
