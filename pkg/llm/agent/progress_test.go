package agent

import (
	"context"
	"testing"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/progress"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

func TestProgressObserverRunIDsDistinct(t *testing.T) {
	var starts []string
	a, err := NewAgent(NewAgentInput{Name: "test", SystemPrompt: "test", Config: llm.CompletionCallInput{
		Client: &fakeChatClient{}, Model: "fake", ProgressObserver: llm.ProgressObserverFunc(func(e llm.ProgressEvent) {
			if e.Kind == llm.ProgressRunStarted {
				starts = append(starts, e.RunID)
			}
		}),
	}})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if _, err := a.CallWithUserMessage("hello"); err != nil {
			t.Fatal(err)
		}
	}
	if len(starts) != 2 || starts[0] == "" || starts[0] == starts[1] {
		t.Fatalf("run IDs=%v", starts)
	}
}

func TestProgressObserverDelegation(t *testing.T) {
	for _, observed := range []bool{false, true} {
		t.Run(map[bool]string{false: "unobserved_child", true: "observed_child"}[observed], func(t *testing.T) {
			var parentEvents, childEvents []llm.ProgressEvent
			inner := llmtool.NewToolbox()
			if err := inner.RegisterTool(llmtool.NewTool("inner", "work", func(ctx context.Context, _ struct{}) (struct{}, error) {
				llm.ReportMilestone(ctx, "child verified")
				return struct{}{}, nil
			})); err != nil {
				t.Fatal(err)
			}
			childConfig := llm.CompletionCallInput{Client: &fakeChatClient{responses: []*llm.ChatResponse{{ToolCalls: []llm.ToolCall{{ID: "inner-call", Name: "inner", Arguments: `{}`}}}, {Content: "child done"}}}, Model: "fake", Tools: *inner}
			if observed {
				childConfig.ProgressObserver = llm.ProgressObserverFunc(func(e llm.ProgressEvent) { childEvents = append(childEvents, e) })
			}
			child, err := NewAgent(NewAgentInput{Name: "child", SystemPrompt: "child", Config: childConfig})
			if err != nil {
				t.Fatal(err)
			}
			outer := llmtool.NewToolbox()
			if err := outer.RegisterTool(child.AsTool()); err != nil {
				t.Fatal(err)
			}
			parent, err := NewAgent(NewAgentInput{Name: "parent", SystemPrompt: "parent", Config: llm.CompletionCallInput{
				Client: &fakeChatClient{responses: []*llm.ChatResponse{{ToolCalls: []llm.ToolCall{{ID: "delegate", Name: "agent_child", Arguments: `{"query":"work"}`}}}, {Content: "parent done"}}}, Model: "fake", Tools: *outer,
				ProgressObserver: llm.ProgressObserverFunc(func(e llm.ProgressEvent) { parentEvents = append(parentEvents, e) }),
			}})
			if err != nil {
				t.Fatal(err)
			}
			out, err := parent.CallWithUserMessage("delegate")
			if err != nil || out.Content != "parent done" {
				t.Fatalf("output=%#v err=%v", out, err)
			}
			for _, event := range parentEvents {
				if event.Kind == llm.ProgressMilestone || event.ToolName == "inner" {
					t.Fatal("child observation leaked into parent")
				}
			}
			if observed {
				if len(childEvents) == 0 || childEvents[0].RunID == parentEvents[0].RunID {
					t.Fatal("child lacks independent run")
				}
				milestones := 0
				for _, event := range childEvents {
					if event.Kind == llm.ProgressMilestone {
						milestones++
						if event.Message != "child verified" || event.ToolCallID != "inner-call" {
							t.Fatal(event)
						}
					}
				}
				if milestones != 1 {
					t.Fatal("missing child milestone")
				}
			}
		})
	}
}

func TestProgressSnapshotDuringAgentRun(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	defer close(release)
	box := llmtool.NewToolbox()
	if err := box.RegisterTool(llmtool.NewTool("wait", "offline work", func(ctx context.Context, _ struct{}) (struct{}, error) {
		llm.ReportMilestone(ctx, "verified before wait")
		close(entered)
		<-release
		return struct{}{}, nil
	})); err != nil {
		t.Fatal(err)
	}
	tracker := progress.NewTracker()
	a, err := NewAgent(NewAgentInput{Name: "test", SystemPrompt: "test", Config: llm.CompletionCallInput{Client: &fakeChatClient{responses: []*llm.ChatResponse{{ToolCalls: []llm.ToolCall{{ID: "wait-call", Name: "wait", Arguments: `{}`}}}, {Content: "done"}}}, Model: "fake", Tools: *box, ProgressObserver: tracker}})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := a.CallWithUserMessage("work"); done <- err }()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("tool did not start")
	}
	s := tracker.Snapshot()
	if s.Status != "running" || len(s.Milestones) != 1 || s.Milestones[0] != "verified before wait" {
		t.Fatal(s)
	}
	// Release once, then wait for completion before checking the terminal view.
	release <- struct{}{}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("agent did not finish")
	}
	if tracker.Snapshot().Status != "succeeded" {
		t.Fatal(tracker.Snapshot())
	}
}
