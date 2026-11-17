// Package progress provides bounded, process-local snapshots of one completion run.
package progress

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

// Config limits retained snapshot data. Zero fields use defaults; negative
// fields are invalid. Limits are copied at construction and cannot be changed.
type Config struct {
	MaxCompletedTools int // Default 100.
	MaxFailedTools    int // Default 100.
	MaxMilestones     int // Default 100.
	MaxStringBytes    int // Default 1024; applies to tool names and milestones.
}

// Snapshot is a detached view. Tool names are internal metadata; milestone
// strings are caller-supplied data. Map/redact them before user-facing messages.
// Before the first run starts, fields have their zero values.
type Snapshot struct {
	RunID          string    `json:"runId"`
	Status         string    `json:"status"`
	Stage          string    `json:"stage"`
	CompletedTools []string  `json:"completedTools"`
	FailedTools    []string  `json:"failedTools"`
	Milestones     []string  `json:"milestones"`
	Version        uint64    `json:"version"`
	UpdatedAt      time.Time `json:"updatedAt"`
}

// Tracker observes one run, binding on its first valid run_started event.
// Other runs, stale events, and all events after termination are ignored.
// Create another tracker for another run. Retention is lossy, not an event log
// or a permanent replay-deduplication store. The zero value uses default limits.
type Tracker struct {
	mu       sync.RWMutex
	config   Config
	snapshot Snapshot
	active   *toolAttempt
}

var _ llm.ProgressObserver = (*Tracker)(nil)

type toolAttempt struct {
	key     [32]byte
	started time.Time
}

func defaultConfig() Config { return Config{100, 100, 100, 1024} }

// NewTracker returns a single-run tracker with default retention limits.
func NewTracker() *Tracker { return &Tracker{config: defaultConfig()} }

// NewTrackerWithConfig validates and copies retention limits.
func NewTrackerWithConfig(config Config) (*Tracker, error) {
	defaults := defaultConfig()
	for _, field := range []struct {
		value    *int
		fallback int
	}{
		{&config.MaxCompletedTools, defaults.MaxCompletedTools},
		{&config.MaxFailedTools, defaults.MaxFailedTools},
		{&config.MaxMilestones, defaults.MaxMilestones},
		{&config.MaxStringBytes, defaults.MaxStringBytes},
	} {
		if *field.value < 0 {
			return nil, errors.New("progress tracker limits cannot be negative")
		}
		if *field.value == 0 {
			*field.value = field.fallback
		}
	}
	return &Tracker{config: config}, nil
}

// OnProgress records meaningful changes with bounded storage. Run IDs must be
// nonempty and at most 1024 bytes (SDK-generated IDs are much shorter). Tool
// outcomes require a matching start; fixed-size identity digests avoid retaining
// potentially large call IDs and distinguish repeated tool names.
func (t *Tracker) OnProgress(event llm.ProgressEvent) {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := &t.snapshot
	if s.Status == "succeeded" || s.Status == "failed" {
		return
	}
	if s.RunID == "" {
		if event.Kind != llm.ProgressRunStarted || event.RunID == "" || len(event.RunID) > 1024 {
			return
		}
		if t.config == (Config{}) {
			t.config = defaultConfig()
		}
		s.RunID, s.Status = strings.Clone(event.RunID), "running"
	} else if event.RunID != s.RunID || event.Kind == llm.ProgressRunStarted || event.Time.Before(s.UpdatedAt) {
		return
	}
	changed := string(event.Kind) != s.Stage
	switch event.Kind {
	case llm.ProgressRunStarted, llm.ProgressGenerationStarted, llm.ProgressGenerationFinished:
	case llm.ProgressToolStarted:
		key := toolKey(event)
		if t.active != nil && t.active.key == key && t.active.started.Equal(event.Time) {
			return
		}
		t.active = &toolAttempt{key: key, started: event.Time}
		changed = true
	case llm.ProgressToolSucceeded, llm.ProgressToolFailed:
		if t.active == nil || t.active.key != toolKey(event) || event.Time.Before(t.active.started) {
			return
		}
		t.active = nil
		name := t.text(event.ToolName)
		if event.Kind == llm.ProgressToolSucceeded {
			s.CompletedTools = retain(s.CompletedTools, name, t.config.MaxCompletedTools)
		} else {
			s.FailedTools = retain(s.FailedTools, name, t.config.MaxFailedTools)
		}
		changed = true
	case llm.ProgressMilestone:
		if strings.TrimSpace(event.Message) == "" {
			return
		}
		message := t.text(event.Message)
		if slices.Contains(s.Milestones, message) {
			return
		}
		s.Milestones = retain(s.Milestones, message, t.config.MaxMilestones)
		changed = true
	case llm.ProgressRunSucceeded:
		s.Status, t.active = "succeeded", nil
		changed = true
	case llm.ProgressRunFailed:
		s.Status, t.active = "failed", nil
		changed = true
	default:
		return
	}
	if changed {
		s.Stage = string(event.Kind)
		s.Version++
		s.UpdatedAt = event.Time.UTC()
	}
}

// Snapshot returns copied slices without acquiring any agent lock.
func (t *Tracker) Snapshot() Snapshot {
	t.mu.RLock()
	defer t.mu.RUnlock()
	s := t.snapshot
	s.CompletedTools = slices.Clone(s.CompletedTools)
	s.FailedTools = slices.Clone(s.FailedTools)
	s.Milestones = slices.Clone(s.Milestones)
	return s
}

func toolKey(event llm.ProgressEvent) [32]byte {
	hash := sha256.New()
	_, _ = fmt.Fprintf(hash, "%d:%d:%d:", event.Round, event.Attempt, len(event.ToolCallID))
	_, _ = hash.Write([]byte(event.ToolCallID))
	_, _ = hash.Write([]byte(event.ToolName))
	var key [32]byte
	copy(key[:], hash.Sum(nil))
	return key
}

func (t *Tracker) text(value string) string {
	value = value[:min(len(value), t.config.MaxStringBytes)]
	// Remove invalid/partial UTF-8, then detach any substring from its source.
	return strings.Clone(strings.ToValidUTF8(value, ""))
}

func retain(values []string, value string, limit int) []string {
	if len(values) == limit {
		copy(values, values[1:])
		values[len(values)-1] = value
		return values
	}
	return append(values, value)
}
