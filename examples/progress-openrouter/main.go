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

const defaultTask = "Compare the demo suppliers for 10 sensors. Research the catalog, calculate delivered totals, then recommend the cheapest available supplier. Clearly label all prices as demo data."

type config struct {
	apiKey, model, baseURL, task string
	timeout, statusInterval      time.Duration
}

func main() {
	cfg := config{apiKey: os.Getenv("OPENROUTER_API_KEY"), model: os.Getenv("OPENROUTER_MODEL"), baseURL: os.Getenv("OPENROUTER_BASE_URL")}
	flag.StringVar(&cfg.task, "task", defaultTask, "task for the coordinator; tools use demo sensor suppliers")
	flag.DurationVar(&cfg.timeout, "timeout", 3*time.Minute, "overall workflow timeout")
	flag.DurationVar(&cfg.statusInterval, "status-interval", time.Second, "interval for changed status snapshots")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	err := run(ctx, cfg, os.Stdout)
	stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

type supplier struct {
	ID             string `json:"id"`
	UnitPriceCents int    `json:"unit_price_cents"`
	ShippingCents  int    `json:"shipping_cents"`
	Stock          int    `json:"stock"`
}

var catalog = []supplier{
	{ID: "north", UnitPriceCents: 2400, ShippingCents: 1200, Stock: 40},
	{ID: "south", UnitPriceCents: 2300, ShippingCents: 2400, Stock: 30},
	{ID: "west", UnitPriceCents: 2500, ShippingCents: 0, Stock: 20},
}

type catalogInput struct {
	Category string `json:"category" description:"Demo catalog category: sensor"`
}

func lookupCatalog(ctx context.Context, input catalogInput) ([]supplier, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.ToLower(strings.TrimSpace(input.Category)) != "sensor" {
		return nil, errors.New("the demo catalog contains only the sensor category")
	}
	llm.ReportMilestone(ctx, "Verified 3 suppliers in the local demo sensor catalog")
	return append([]supplier(nil), catalog...), nil
}

type quoteInput struct {
	Quantity int `json:"quantity" description:"Number of demo sensors, from 1 to 100"`
}

type quote struct {
	SupplierID string `json:"supplier_id"`
	TotalCents int    `json:"total_cents"`
}

func calculateQuotes(ctx context.Context, input quoteInput) ([]quote, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if input.Quantity < 1 || input.Quantity > 100 {
		return nil, errors.New("quantity must be between 1 and 100")
	}
	var quotes []quote
	for _, offer := range catalog {
		if offer.Stock >= input.Quantity {
			quotes = append(quotes, quote{SupplierID: offer.ID, TotalCents: input.Quantity*offer.UnitPriceCents + offer.ShippingCents})
		}
	}
	llm.ReportMilestone(ctx, fmt.Sprintf("Calculated %d available demo quotes for %d sensors", len(quotes), input.Quantity))
	return quotes, nil
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

func newCoordinator(client llm.ChatClient, model string, tracking *runTracking) (*agent.Agent, error) {
	researcher, err := newAgent(client, model, tracking, "researcher",
		"You are the researcher. Call lookup_catalog with category sensor to verify the local demo suppliers, prices in cents, and stock. Report only tool-backed facts and label them demo data.",
		llmtool.NewTool("lookup_catalog", "Reads the local demo sensor supplier catalog; no live prices.", lookupCatalog))
	if err != nil {
		return nil, err
	}
	analyst, err := newAgent(client, model, tracking, "analyst",
		"You are the analyst. Call calculate_quotes for the requested quantity. Compare returned delivered totals in cents and recommend the cheapest available option. Label all prices demo data; never invent quotes or availability.",
		llmtool.NewTool("calculate_quotes", "Computes demo delivered quotes for suppliers with enough stock.", calculateQuotes))
	if err != nil {
		return nil, err
	}
	return newAgent(client, model, tracking, "coordinator",
		"You are the coordinator. Complete the user's demo sensor procurement task. First delegate catalog research to agent_researcher, then pass the findings and requested quantity to agent_analyst for verified calculations. Only then give a concise recommendation with delivered total and caveats. All business data is local demo data, not live supplier information.",
		researcher.AsTool(), analyst.AsTool())
}

type namedSnapshot struct {
	Agent string `json:"agent"`
	progress.Snapshot
}

// Keep only the latest run for each fixed role; each run still owns its tracker.
type runTracking struct {
	mu      sync.Mutex
	latest  map[string]*progress.Tracker
	changed chan struct{}
}

func newRunTracking() *runTracking {
	return &runTracking{latest: make(map[string]*progress.Tracker), changed: make(chan struct{}, 1)}
}

func (t *runTracking) observer(name string) llm.ProgressObserver {
	return llm.ProgressObserverFunc(func(event llm.ProgressEvent) {
		t.mu.Lock()
		tracker := t.latest[name]
		if event.Kind == llm.ProgressRunStarted && (tracker == nil || tracker.Snapshot().RunID != event.RunID) {
			tracker = progress.NewTracker()
			t.latest[name] = tracker
		}
		if tracker != nil {
			tracker.OnProgress(event)
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
		if tracker := t.latest[name]; tracker != nil {
			snapshots = append(snapshots, namedSnapshot{Agent: name, Snapshot: tracker.Snapshot()})
		}
	}
	return snapshots
}

func (t *runTracking) printChanges(out io.Writer, previous map[string]progress.Snapshot) error {
	for _, snapshot := range t.snapshots() {
		last := previous[snapshot.Agent]
		if last.RunID == snapshot.RunID && last.Version == snapshot.Version {
			continue
		}
		if _, err := fmt.Fprintf(out, "STATUS %-11s run=%s status=%s stage=%s tools=%d/%d milestones=%q\n",
			snapshot.Agent, snapshot.RunID, snapshot.Status, snapshot.Stage,
			len(snapshot.CompletedTools), len(snapshot.FailedTools), snapshot.Milestones); err != nil {
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
	client, _, err := openrouter.NewClient(openrouter.ClientConfig{
		APIKey: cfg.apiKey, BaseURL: cfg.baseURL, Timeout: time.Minute,
		Provider: openrouter.ProviderPreferences{ExtraFields: map[string]any{"require_parameters": true}},
	})
	if err != nil {
		return err
	}
	tracking := newRunTracking()
	coordinator, err := newCoordinator(client, cfg.model, tracking)
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
		select {
		case <-tracking.changed:
		case <-ticker.C:
		case result := <-done:
			printErr := tracking.printChanges(out, previous)
			if result.err == nil {
				_, err = fmt.Fprintf(out, "\nFINAL ANSWER (demo supplier data)\n%s\n", result.output.Content)
			}
			_, headingErr := fmt.Fprintln(out, "\nFINAL SNAPSHOTS (independent runs; latest per role)")
			encoder := json.NewEncoder(out)
			encoder.SetIndent("", "  ")
			return errors.Join(result.err, printErr, err, headingErr, encoder.Encode(tracking.snapshots()))
		}
		if err := tracking.printChanges(out, previous); err != nil {
			cancel()
			result := <-done
			return errors.Join(err, result.err)
		}
	}
}
