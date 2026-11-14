// This example uses OpenRouter for planning and local fixtures for business data.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/agent"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/progress"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/provider/openrouter"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

const defaultTask = "Plan a demo order for 10 sensors with a 7-day delivery deadline. Research catalog, stock, and delivery; calculate delivered quotes; compare quantity scenarios 10, 20, and 30 under that deadline; audit the recommended supplier and total for the primary 10-unit order. Present verified findings, scenario tradeoffs, and the audit result. Clearly label all prices and availability as local demo data."

type config struct {
	apiKey, model, baseURL, task       string
	timeout, statusInterval, stepDelay time.Duration
}

func main() {
	cfg := config{apiKey: os.Getenv("OPENROUTER_API_KEY"), model: os.Getenv("OPENROUTER_MODEL"), baseURL: os.Getenv("OPENROUTER_BASE_URL")}
	flag.StringVar(&cfg.task, "task", defaultTask, "task for the coordinator; tools use demo sensor suppliers")
	flag.DurationVar(&cfg.timeout, "timeout", 3*time.Minute, "overall workflow timeout")
	flag.DurationVar(&cfg.statusInterval, "status-interval", time.Second, "interval for changed status snapshots")
	flag.DurationVar(&cfg.stepDelay, "step-delay", 2*time.Second, "simulated latency per demo check; 0s for fast runs")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	err := run(ctx, cfg, os.Stdout)
	stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func newAgent(client llm.ChatClient, model string, tracking *runTracking, name, prompt string, tools ...llmtool.Tool) (*agent.Agent, error) {
	box := llmtool.NewToolbox()
	for _, tool := range tools {
		if err := box.RegisterTool(tool); err != nil {
			return nil, err
		}
	}
	return agent.NewAgent(agent.NewAgentInput{
		Name: name, Description: prompt, SystemPrompt: prompt,
		Config: llm.CompletionCallInput{
			Client: client, Model: model, Tools: *box, ProgressObserver: tracking.observer(name),
			MaxToolCallRounds: 4, MaxProviderAttempts: 8, MaxToolCalls: 6,
		},
	})
}

func newCoordinator(client llm.ChatClient, model string, tracking *runTracking, stepDelay time.Duration) (*agent.Agent, error) {
	demo := demoTools{delay: stepDelay}
	researcher, err := newAgent(client, model, tracking, "researcher",
		"You are the researcher. In order, call lookup_catalog with category sensor, check_inventory for the primary order quantity, and check_delivery for the requested deadline. All three checks are required. Report only tool-backed supplier, stock, and delivery facts and label them local demo data.",
		llmtool.NewTool("lookup_catalog", "Reads the local demo sensor catalog.", demo.lookupCatalog),
		llmtool.NewTool("check_inventory", "Checks demo stock separately for every supplier.", demo.checkInventory),
		llmtool.NewTool("check_delivery", "Checks each demo supplier against a delivery deadline.", demo.checkDelivery))
	if err != nil {
		return nil, err
	}
	analyst, err := newAgent(client, model, tracking, "analyst",
		"You are the analyst. In order, call calculate_quotes for the primary quantity, evaluate_scenarios for the requested distinct quantities and delivery deadline, then audit_recommendation for your chosen primary-order supplier and exact total. All three tools are required before your final answer. Choose using verified stock, deadline, and total facts; if an audit is not approved, correct the choice and audit again. Label all data demo fixtures, never live quotes.",
		llmtool.NewTool("calculate_quotes", "Calculates each demo supplier's quote or stock exclusion.", demo.calculateQuotes),
		llmtool.NewTool("evaluate_scenarios", "Compares up to five quantity scenarios under a delivery deadline.", demo.evaluateScenarios),
		llmtool.NewTool("audit_recommendation", "Audits demo stock, delivery, proposed total, and cheapest eligible selection.", demo.auditRecommendation))
	if err != nil {
		return nil, err
	}
	return newAgent(client, model, tracking, "coordinator",
		"You are the coordinator. First delegate catalog, inventory, and delivery research to agent_researcher. Then pass the findings, primary quantity, scenario quantities, and deadline to agent_analyst for quotes, scenarios, and a recommendation audit. Only after both agents finish, give a concise report with the approved primary recommendation, scenario tradeoffs, and caveats. Never claim an unaudited recommendation is approved. All business data is local demo fixtures, not live supplier information.",
		researcher.AsTool(), analyst.AsTool())
}

type namedSnapshot struct {
	Agent   string `json:"agent"`
	Elapsed string `json:"elapsed"`
	progress.Snapshot
}

type trackedRun struct {
	tracker           *progress.Tracker
	started, finished time.Time
}

// Keep only the latest run for each fixed role; each run still owns its tracker.
type runTracking struct {
	mu      sync.Mutex
	latest  map[string]*trackedRun
	changed chan struct{}
}

func newRunTracking() *runTracking {
	return &runTracking{latest: make(map[string]*trackedRun), changed: make(chan struct{}, 1)}
}

func (t *runTracking) observer(name string) llm.ProgressObserver {
	return llm.ProgressObserverFunc(func(event llm.ProgressEvent) {
		t.mu.Lock()
		run := t.latest[name]
		if event.Kind == llm.ProgressRunStarted && (run == nil || run.tracker.Snapshot().RunID != event.RunID) {
			run = &trackedRun{tracker: progress.NewTracker(), started: time.Now()}
			t.latest[name] = run
		}
		if run != nil {
			run.tracker.OnProgress(event)
			if (event.Kind == llm.ProgressRunSucceeded || event.Kind == llm.ProgressRunFailed) && run.finished.IsZero() && run.tracker.Snapshot().RunID == event.RunID {
				run.finished = time.Now()
			}
		}
		t.mu.Unlock()
		// The callback only records state and signals; the status loop owns output.
		switch event.Kind {
		case llm.ProgressRunStarted, llm.ProgressMilestone, llm.ProgressRunSucceeded, llm.ProgressRunFailed:
			select {
			case t.changed <- struct{}{}:
			default:
			}
		}
	})
}

func (t *runTracking) snapshots() []namedSnapshot {
	t.mu.Lock()
	defer t.mu.Unlock()
	var snapshots []namedSnapshot
	for _, name := range []string{"coordinator", "researcher", "analyst"} {
		if run := t.latest[name]; run != nil {
			end := run.finished
			if end.IsZero() {
				end = time.Now()
			}
			snapshots = append(snapshots, namedSnapshot{Agent: name, Elapsed: end.Sub(run.started).Round(time.Second).String(), Snapshot: run.tracker.Snapshot()})
		}
	}
	return snapshots
}

func (t *runTracking) printChanges(out io.Writer, previous map[string]progress.Snapshot, heartbeat bool) error {
	for _, snapshot := range t.snapshots() {
		last := previous[snapshot.Agent]
		if last.RunID == snapshot.RunID && last.Version == snapshot.Version {
			if heartbeat && snapshot.Status == "running" {
				if _, err := fmt.Fprintf(out, "HEARTBEAT %-11s run=%s elapsed=%s still_running stage=%s version=%d\n", snapshot.Agent, snapshot.RunID, snapshot.Elapsed, snapshot.Stage, snapshot.Version); err != nil {
					return err
				}
			}
			continue
		}
		latest := ""
		if len(snapshot.Milestones) > 0 {
			latest = snapshot.Milestones[len(snapshot.Milestones)-1]
		}
		if _, err := fmt.Fprintf(out, "STATUS %-11s run=%s elapsed=%s status=%s stage=%s tools=%d/%d milestones=%d latest=%q version=%d\n",
			snapshot.Agent, snapshot.RunID, snapshot.Elapsed, snapshot.Status, snapshot.Stage,
			len(snapshot.CompletedTools), len(snapshot.FailedTools), len(snapshot.Milestones), latest, snapshot.Version); err != nil {
			return err
		}
		previous[snapshot.Agent] = snapshot.Snapshot
	}
	return nil
}

func run(ctx context.Context, cfg config, out io.Writer) error {
	if strings.TrimSpace(cfg.model) == "" || strings.TrimSpace(cfg.apiKey) == "" {
		return errors.New("set OPENROUTER_API_KEY and OPENROUTER_MODEL to a tool-capable model")
	}
	if strings.TrimSpace(cfg.task) == "" || cfg.timeout <= 0 || cfg.statusInterval <= 0 {
		return errors.New("task must be nonblank; timeout and status interval must be positive")
	}
	if cfg.stepDelay < 0 {
		return errors.New("step delay cannot be negative")
	}
	client, _, err := openrouter.NewClient(openrouter.ClientConfig{
		APIKey: cfg.apiKey, BaseURL: cfg.baseURL, Timeout: time.Minute,
		Provider: openrouter.ProviderPreferences{ExtraFields: map[string]any{"require_parameters": true}},
	})
	if err != nil {
		return err
	}
	tracking := newRunTracking()
	coordinator, err := newCoordinator(client, cfg.model, tracking, cfg.stepDelay)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, cfg.timeout)
	defer cancel()
	type result struct {
		output *llm.CompletionCallOutput
		err    error
	}
	done := make(chan result, 1)
	go func() {
		output, err := coordinator.CallWithUserMessageContext(ctx, cfg.task)
		done <- result{output, err}
	}()
	ticker := time.NewTicker(cfg.statusInterval)
	defer ticker.Stop()
	previous := make(map[string]progress.Snapshot)
	for {
		heartbeat := false
		select {
		case <-tracking.changed:
		case <-ticker.C:
			heartbeat = true
		case result := <-done:
			printErr := tracking.printChanges(out, previous, false)
			if result.err == nil {
				_, err = fmt.Fprintf(out, "\nFINAL ANSWER (demo supplier data)\n%s\n", result.output.Content)
			}
			_, headingErr := fmt.Fprintln(out, "\nFINAL SNAPSHOTS (independent runs; latest per role)")
			encoder := json.NewEncoder(out)
			encoder.SetIndent("", "  ")
			return errors.Join(result.err, printErr, err, headingErr, encoder.Encode(tracking.snapshots()))
		}
		if err := tracking.printChanges(out, previous, heartbeat); err != nil {
			cancel()
			result := <-done
			return errors.Join(err, result.err)
		}
	}
}
