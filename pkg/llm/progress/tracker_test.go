package progress

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

func event(kind llm.ProgressKind, tick int) llm.ProgressEvent {
	return llm.ProgressEvent{RunID: "run-1", Kind: kind, Time: time.Unix(int64(tick), 0).UTC(), ToolCallID: "call", ToolName: "work"}
}

func TestTrackerOwnershipVersionsAndCopies(t *testing.T) {
	tracker := NewTracker()
	tracker.OnProgress(event(llm.ProgressToolStarted, 0))
	if tracker.Snapshot().Version != 0 {
		t.Fatal("accepted event before start")
	}
	tracker.OnProgress(event(llm.ProgressRunStarted, 1))
	started := event(llm.ProgressToolStarted, 2)
	tracker.OnProgress(started)
	version := tracker.Snapshot().Version
	tracker.OnProgress(started)
	if tracker.Snapshot().Version != version {
		t.Fatal("duplicate start changed version")
	}
	tracker.OnProgress(event(llm.ProgressToolSucceeded, 3))
	tracker.OnProgress(event(llm.ProgressToolSucceeded, 3))
	tracker.OnProgress(event(llm.ProgressToolStarted, 4))
	tracker.OnProgress(event(llm.ProgressToolFailed, 5))
	milestone := event(llm.ProgressMilestone, 6)
	milestone.Message = "verified"
	tracker.OnProgress(milestone)
	before := tracker.Snapshot()
	tracker.OnProgress(milestone)
	foreign := event(llm.ProgressRunFailed, 7)
	foreign.RunID = "foreign"
	tracker.OnProgress(foreign)
	tracker.OnProgress(event(llm.ProgressGenerationStarted, 1))
	if !reflect.DeepEqual(tracker.Snapshot(), before) {
		t.Fatal("duplicate, foreign or stale event changed state")
	}
	copy := tracker.Snapshot()
	copy.CompletedTools[0] = "changed"
	copy.FailedTools[0] = "changed"
	copy.Milestones[0] = "changed"
	if !reflect.DeepEqual(tracker.Snapshot(), before) {
		t.Fatal("snapshot aliases tracker slices")
	}
	tracker.OnProgress(event(llm.ProgressRunSucceeded, 8))
	terminal := tracker.Snapshot()
	for _, kind := range []llm.ProgressKind{llm.ProgressRunStarted, llm.ProgressToolStarted, llm.ProgressRunFailed, llm.ProgressMilestone} {
		next := event(kind, 9)
		next.Message = "late"
		tracker.OnProgress(next)
	}
	nextRun := event(llm.ProgressRunStarted, 10)
	nextRun.RunID = "next"
	tracker.OnProgress(nextRun)
	if terminal.Status != "succeeded" || !reflect.DeepEqual(tracker.Snapshot(), terminal) {
		t.Fatal("terminal state changed")
	}
}

func TestTrackerConfiguredBoundsAndRepeatedIDs(t *testing.T) {
	config := Config{MaxCompletedTools: 2, MaxFailedTools: 3, MaxMilestones: 2, MaxStringBytes: 5}
	tracker, err := NewTrackerWithConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	config.MaxCompletedTools = 999
	tracker.OnProgress(event(llm.ProgressRunStarted, 1))
	for i := 0; i < 2000; i++ {
		start := event(llm.ProgressToolStarted, 2+3*i)
		// Identical or empty IDs remain distinct when a new start/end pair occurs.
		if i%2 == 0 {
			start.ToolCallID = ""
		} else {
			start.ToolCallID = strings.Repeat("large-id", 1000)
		}
		start.ToolName = fmt.Sprintf("%d🙂large", i)
		tracker.OnProgress(start)
		end := start
		end.Time = end.Time.Add(time.Second)
		end.Kind = llm.ProgressToolSucceeded
		if i%2 == 0 {
			end.Kind = llm.ProgressToolFailed
		}
		tracker.OnProgress(end)
		milestone := event(llm.ProgressMilestone, 4+3*i)
		milestone.Message = fmt.Sprintf("%d🙂verified", i)
		tracker.OnProgress(milestone)
	}
	s := tracker.Snapshot()
	if len(s.CompletedTools) != 2 || len(s.FailedTools) != 3 || len(s.Milestones) != 2 || s.CompletedTools[1] != "1999" || s.FailedTools[2] != "1998" || s.Milestones[1] != "1999" {
		t.Fatalf("retention=%#v", s)
	}
	for _, values := range [][]string{s.CompletedTools, s.FailedTools, s.Milestones} {
		for _, value := range values {
			if len(value) > 5 || !utf8.ValidString(value) {
				t.Fatalf("invalid bounded string %q", value)
			}
		}
	}
	if tracker.active != nil || tracker.config.MaxCompletedTools != 2 {
		t.Fatal("retained active identity or mutable config")
	}
}

func TestTrackerConfigValidationAndUTF8(t *testing.T) {
	defaults, err := NewTrackerWithConfig(Config{})
	if err != nil || defaults.config != NewTracker().config {
		t.Fatalf("default config=%#v err=%v", defaults, err)
	}
	for _, config := range []Config{{MaxCompletedTools: -1}, {MaxFailedTools: -1}, {MaxMilestones: -1}, {MaxStringBytes: -1}} {
		if got, err := NewTrackerWithConfig(config); err == nil || got != nil {
			t.Fatal("accepted negative config")
		}
	}
	tracker, err := NewTrackerWithConfig(Config{MaxMilestones: 1, MaxStringBytes: 4})
	if err != nil {
		t.Fatal(err)
	}
	tracker.OnProgress(event(llm.ProgressRunStarted, 1))
	for i, message := range []string{"ééé", "🙂more", "a🙂more"} {
		e := event(llm.ProgressMilestone, i+2)
		e.Message = message
		tracker.OnProgress(e)
		s := tracker.Snapshot()
		if len(s.Milestones) != 1 || len(s.Milestones[0]) > 4 || !utf8.ValidString(s.Milestones[0]) {
			t.Fatal(s)
		}
	}
	if tracker.Snapshot().Milestones[0] != "a" {
		t.Fatal("partial rune retained")
	}
	for _, id := range []string{"", strings.Repeat("x", 1025)} {
		var zero Tracker
		e := event(llm.ProgressRunStarted, 1)
		e.RunID = id
		zero.OnProgress(e)
		if zero.Snapshot().RunID != "" {
			t.Fatal("accepted invalid run ID")
		}
	}
	var zero Tracker
	zero.OnProgress(event(llm.ProgressRunStarted, 1))
	if zero.Snapshot().Status != "running" {
		t.Fatal("zero tracker unusable")
	}
}

func TestTrackerConcurrentSnapshots(t *testing.T) {
	tracker := NewTracker()
	tracker.OnProgress(event(llm.ProgressRunStarted, 1))
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var version uint64
			for j := 0; j < 2000; j++ {
				s := tracker.Snapshot()
				if s.Version < version {
					t.Error("version regressed")
				}
				version = s.Version
				for k := range s.Milestones {
					s.Milestones[k] = "reader copy"
				}
				tracker.OnProgress(event(llm.ProgressGenerationStarted, 0))
			}
		}()
	}
	for i := 0; i < 1000; i++ {
		e := event(llm.ProgressMilestone, i+2)
		e.Message = fmt.Sprint(i)
		tracker.OnProgress(e)
	}
	tracker.OnProgress(event(llm.ProgressRunFailed, 1002))
	wg.Wait()
	s := tracker.Snapshot()
	if s.Status != "failed" || len(s.Milestones) != 100 || s.Milestones[99] != "999" {
		t.Fatal(s)
	}
}
