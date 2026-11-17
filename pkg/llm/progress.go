package llm

import (
	"context"
	"crypto/rand"
	"errors"
	"strings"
	"sync"
	"time"

	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

// ProgressKind identifies a coarse completion lifecycle transition.
type ProgressKind string

const (
	ProgressRunStarted         ProgressKind = "run_started"
	ProgressGenerationStarted  ProgressKind = "generation_started"
	ProgressGenerationFinished ProgressKind = "generation_finished"
	ProgressToolStarted        ProgressKind = "tool_started"
	ProgressToolSucceeded      ProgressKind = "tool_succeeded"
	ProgressToolFailed         ProgressKind = "tool_failed"
	ProgressMilestone          ProgressKind = "milestone"
	ProgressRunSucceeded       ProgressKind = "run_succeeded"
	ProgressRunFailed          ProgressKind = "run_failed"
)

// ProgressEvent contains lifecycle metadata, never automatic prompt, reasoning,
// argument, result, or raw error text. Round and Attempt are zero-based.
type ProgressEvent struct {
	RunID      string
	Kind       ProgressKind
	Time       time.Time
	Round      int
	Attempt    int
	ToolCallID string
	ToolName   string
	Message    string // Caller-supplied, verified milestone text only.
	ErrorKind  string // Fixed category; empty for success.
}

// ProgressObserver observes execution without controlling it. Callbacks are
// synchronous and serialized per run; return quickly, avoid network work, and
// do not recursively report progress. Shared observers must support multiple
// concurrent runs. Callback panics are isolated from completion execution.
type ProgressObserver interface {
	OnProgress(ProgressEvent)
}

// ProgressObserverFunc adapts a function to a ProgressObserver.
type ProgressObserverFunc func(ProgressEvent)

func (f ProgressObserverFunc) OnProgress(event ProgressEvent) {
	if f != nil {
		f(event)
	}
}

// ReportMilestone reports a verified accomplishment from an active tool
// context. Text is trusted application data: callers must avoid secrets and
// unverified claims. Missing observation, blank text, and closed scopes are no-ops.
func ReportMilestone(ctx context.Context, message string) {
	if ctx == nil || strings.TrimSpace(message) == "" {
		return
	}
	span, _ := ctx.Value(progressContextKey{}).(*progressSpan)
	if span == nil {
		return
	}
	p := span.run
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed || !span.milestonesOpen {
		return
	}
	event := span.event
	event.Kind = ProgressMilestone
	event.Message = message
	p.emitLocked(event)
}

type progressContextKey struct{}

type progressSpan struct {
	run            *runProgress
	event          ProgressEvent
	milestonesOpen bool // Guarded by run.mu.
	failure        string
}

type runProgress struct {
	mu         sync.Mutex
	observer   ProgressObserver
	id         string
	round      int
	attempt    int
	fallback   string
	closed     bool
	generation *progressSpan
	tool       *progressSpan
}

func startRunProgress(observer ProgressObserver) *runProgress {
	if observer == nil {
		return nil
	}
	p := &runProgress{observer: observer, id: rand.Text(), fallback: "provider_error"}
	p.emitLocked(ProgressEvent{Kind: ProgressRunStarted})
	return p
}

func (p *runProgress) emitLocked(event ProgressEvent) {
	event.RunID = p.id
	event.Time = time.Now().UTC()
	func() {
		defer func() { _ = recover() }()
		p.observer.OnProgress(event)
	}()
}

// finish is deferred directly so recover observes an original execution panic.
func (p *runProgress) finish(err *error) {
	if p == nil {
		return
	}
	panicked := recover()
	p.mu.Lock()
	kind := progressErrorKind(*err, p.fallback)
	if panicked != nil {
		kind = "panic"
	}
	if p.tool != nil {
		p.finishToolLocked(kind)
	}
	if p.generation != nil {
		p.finishGenerationLocked(kind)
	}
	p.closed = true
	event := ProgressEvent{Kind: ProgressRunSucceeded, Round: p.round, Attempt: p.attempt}
	if kind != "" {
		event.Kind, event.ErrorKind = ProgressRunFailed, kind
	}
	p.emitLocked(event)
	p.mu.Unlock()
	if panicked != nil {
		panic(panicked)
	}
}

func (p *runProgress) startGeneration(round, attempt int) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.round, p.attempt, p.fallback = round, attempt, "provider_error"
	p.generation = &progressSpan{run: p, event: ProgressEvent{
		Kind: ProgressGenerationStarted, Round: round, Attempt: attempt,
	}}
	p.emitLocked(p.generation.event)
}

func (p *runProgress) finishGeneration(err error, nilResponse bool) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	kind := progressErrorKind(err, "provider_error")
	if err == nil && nilResponse {
		kind = "invalid_result"
	}
	if kind != "" {
		p.fallback = kind
	}
	p.finishGenerationLocked(kind)
}

func (p *runProgress) finishGenerationLocked(kind string) {
	event := p.generation.event
	event.Kind, event.ErrorKind = ProgressGenerationFinished, kind
	p.generation = nil
	p.emitLocked(event)
}

func (p *runProgress) startTool(ctx context.Context, call ToolCall) context.Context {
	if p == nil {
		return ctx
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.fallback = "tool_error"
	p.tool = &progressSpan{run: p, milestonesOpen: true, event: ProgressEvent{
		Kind: ProgressToolStarted, Round: p.round, Attempt: p.attempt,
		ToolCallID: call.ID, ToolName: call.Name,
	}}
	p.emitLocked(p.tool.event)
	return context.WithValue(ctx, progressContextKey{}, p.tool)
}

func (p *runProgress) toolReturned(err error) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tool.milestonesOpen = false
	p.tool.failure = progressErrorKind(err, "tool_error")
}

func (p *runProgress) invalidToolResult() {
	if p != nil {
		p.mu.Lock()
		p.tool.failure, p.fallback = "invalid_result", "invalid_result"
		p.mu.Unlock()
	}
}

func (p *runProgress) finishTool() {
	if p != nil {
		p.mu.Lock()
		p.finishToolLocked("")
		p.mu.Unlock()
	}
}

func (p *runProgress) finishToolLocked(kind string) {
	span := p.tool
	span.milestonesOpen = false
	if kind == "" {
		kind = span.failure
	}
	event := span.event
	event.Kind = ProgressToolSucceeded
	if kind != "" {
		event.Kind, event.ErrorKind = ProgressToolFailed, kind
	}
	p.tool = nil
	p.emitLocked(event)
}

func progressErrorKind(err error, fallback string) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "deadline_exceeded"
	case errors.Is(err, ErrMessageGuardrailBlocked):
		return "guardrail_blocked"
	case errors.Is(err, ErrCompletionEventAborted):
		return "hook_aborted"
	case errors.Is(err, llmtool.ErrToolPolicyDenied):
		return "policy_denied"
	case errors.Is(err, llmtool.ErrApprovalDenied):
		return "approval_denied"
	case errors.Is(err, ErrCompletionUsageUnavailable):
		return "usage_unavailable"
	case errors.Is(err, ErrCompletionLimitExceeded), errors.Is(err, ErrMaxToolCallRoundsExceeded):
		return "limit_exceeded"
	default:
		return fallback
	}
}
