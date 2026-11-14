// Progress demonstrates live snapshots without credentials or network calls.
package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/agent"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/progress"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

type fakeClient struct{}

func (fakeClient) ListModels(context.Context) ([]string, error) {
	return []string{"offline-model"}, nil
}

func (fakeClient) Complete(_ context.Context, request llm.ChatRequest, _ llm.CompletionHooks) (*llm.ChatResponse, error) {
	if request.Messages[len(request.Messages)-1].Type == llm.Tool {
		return &llm.ChatResponse{Content: "Research is complete."}, nil
	}
	return &llm.ChatResponse{ToolCalls: []llm.ToolCall{{
		ID: "call-1", Name: "search", Arguments: `{"query":"pricing"}`,
	}}}, nil
}

func main() {
	entered, release := make(chan struct{}), make(chan struct{})
	tools := llmtool.NewToolbox()
	err := tools.RegisterTool(llmtool.NewTool("search", "Simulated search", func(ctx context.Context, _ struct {
		Query string `json:"query"`
	}) (struct {
		Count int `json:"count"`
	}, error) {
		llm.ReportMilestone(ctx, "Found 3 matching records")
		close(entered)
		<-release
		return struct {
			Count int `json:"count"`
		}{Count: 3}, nil
	}))
	if err != nil {
		panic(err)
	}
	tracker := progress.NewTracker()
	a, err := agent.NewAgent(agent.NewAgentInput{
		Name: "researcher", SystemPrompt: "Use search when useful.",
		Config: llm.CompletionCallInput{
			Client: fakeClient{}, Model: "offline-model", Tools: *tools,
			ProgressObserver: llm.ProgressObserverFunc(func(event llm.ProgressEvent) {
				tracker.OnProgress(event)
				fmt.Printf("%-24s round=%d attempt=%d", event.Kind, event.Round, event.Attempt)
				if event.ToolName != "" {
					fmt.Printf(" tool=%s/%s", event.ToolName, event.ToolCallID)
				}
				if event.Message != "" {
					fmt.Printf(" milestone=%q", event.Message)
				}
				fmt.Println()
			}),
		},
	})
	if err != nil {
		panic(err)
	}
	done := make(chan error, 1)
	var result *llm.CompletionCallOutput
	go func() {
		var runErr error
		result, runErr = a.CallWithUserMessageContext(context.Background(), "Research competitor pricing")
		done <- runErr
	}()
	<-entered
	live := tracker.Snapshot() // No agent lock or status request to the busy agent.
	fmt.Printf("STATUS during work: %s, stage=%s, milestones=%v\n", live.Status, live.Stage, live.Milestones)
	close(release)
	if err := <-done; err != nil {
		panic(err)
	}
	fmt.Println(result.Content)
	data, err := json.MarshalIndent(tracker.Snapshot(), "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(data))
}
