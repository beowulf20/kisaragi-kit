package tool

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ApprovalMode controls whether a tool call needs approval before execution.
type ApprovalMode string

const (
	// ApprovalNever runs the tool without asking for approval.
	ApprovalNever ApprovalMode = "never"
	// ApprovalOnRisk asks for approval when the policy carries a non-low risk.
	ApprovalOnRisk ApprovalMode = "on_risk"
	// ApprovalAlways asks for approval before every call.
	ApprovalAlways ApprovalMode = "always"
)

// RiskLevel describes the declared side-effect risk of a tool.
type RiskLevel string

const (
	RiskLow    RiskLevel = "low"
	RiskMedium RiskLevel = "medium"
	RiskHigh   RiskLevel = "high"
)

// PreviewKind describes the kind of preview the approval UI can show.
type PreviewKind string

const (
	PreviewNone    PreviewKind = "none"
	PreviewPayload PreviewKind = "payload"
	PreviewDiff    PreviewKind = "diff"
	PreviewCommand PreviewKind = "command"
)

// ApprovalPolicy is tool-defined metadata used by the toolbox approval hook.
type ApprovalPolicy struct {
	Mode        ApprovalMode
	Risk        RiskLevel
	Preview     PreviewKind
	Description string
}

// ApprovalRequest is passed to an approval hook before the tool handler runs.
type ApprovalRequest struct {
	ToolCallID       string
	ToolName         string
	Description      string
	Arguments        json.RawMessage
	Policy           ApprovalPolicy
	RequiredByPolicy bool
	PolicyReason     string
	Round            int
	Model            string
}

// ApprovalDecision is returned by an approval hook.
type ApprovalDecision struct {
	Approved bool
	Reason   string
}

// ApprovalRecord records the approval decision for one tool call.
type ApprovalRecord struct {
	ToolCallID       string
	ToolName         string
	Arguments        json.RawMessage
	Policy           ApprovalPolicy
	RequiredByPolicy bool
	PolicyReason     string
	Approved         bool
	Reason           string
	Round            int
	Model            string
}

// ApprovalHook decides whether a tool call may execute.
type ApprovalHook func(context.Context, ApprovalRequest) (ApprovalDecision, error)

var ErrApprovalDenied = errors.New("tool approval denied")

func (policy ApprovalPolicy) requiresApproval() bool {
	switch policy.Mode {
	case ApprovalAlways:
		return true
	case ApprovalOnRisk:
		return policy.Risk == RiskMedium || policy.Risk == RiskHigh
	default:
		return false
	}
}

func approveToolCall(ctx context.Context, hook ApprovalHook, request ToolCallRequest, tool Tool, args json.RawMessage, required bool, requiredByPolicy bool, policyReason string) (*ApprovalRecord, error) {
	if !required {
		return nil, nil
	}

	record := &ApprovalRecord{
		ToolCallID:       request.ID,
		ToolName:         tool.Name,
		Arguments:        append(json.RawMessage(nil), args...),
		Policy:           tool.Approval,
		RequiredByPolicy: requiredByPolicy,
		PolicyReason:     policyReason,
		Round:            request.Round,
		Model:            request.Model,
	}
	if hook == nil {
		record.Reason = "approval hook missing"
		return record, fmt.Errorf("tool %q requires approval: %w", tool.Name, ErrApprovalDenied)
	}

	decision, err := hook(ctx, ApprovalRequest{
		ToolCallID:       request.ID,
		ToolName:         tool.Name,
		Description:      tool.Description,
		Arguments:        args,
		Policy:           tool.Approval,
		RequiredByPolicy: requiredByPolicy,
		PolicyReason:     policyReason,
		Round:            request.Round,
		Model:            request.Model,
	})
	if err != nil {
		record.Reason = err.Error()
		return record, err
	}
	record.Approved = decision.Approved
	record.Reason = decision.Reason
	if !decision.Approved {
		if decision.Reason == "" {
			return record, fmt.Errorf("tool %q: %w", tool.Name, ErrApprovalDenied)
		}
		return record, fmt.Errorf("tool %q: %w: %s", tool.Name, ErrApprovalDenied, decision.Reason)
	}
	return record, nil
}

func (policy ApprovalPolicy) validate() error {
	switch policy.Mode {
	case "", ApprovalNever, ApprovalOnRisk, ApprovalAlways:
	default:
		return fmt.Errorf("invalid approval mode %q", policy.Mode)
	}
	switch policy.Risk {
	case "", RiskLow, RiskMedium, RiskHigh:
	default:
		return fmt.Errorf("invalid approval risk %q", policy.Risk)
	}
	switch policy.Preview {
	case "", PreviewNone, PreviewPayload, PreviewDiff, PreviewCommand:
	default:
		return fmt.Errorf("invalid approval preview %q", policy.Preview)
	}
	return nil
}

// NewStdioApprovalHook asks for human approval over stdin/stdout.
func NewStdioApprovalHook(in io.Reader, out io.Writer) ApprovalHook {
	reader := bufio.NewReader(in)
	return func(_ context.Context, request ApprovalRequest) (ApprovalDecision, error) {
		fmt.Fprintf(out, "\nTool approval required\n")
		fmt.Fprintf(out, "Tool: %s\n", request.ToolName)
		if request.Policy.Description != "" {
			fmt.Fprintf(out, "Intent: %s\n", request.Policy.Description)
		} else if request.Description != "" {
			fmt.Fprintf(out, "Intent: %s\n", request.Description)
		}
		if request.RequiredByPolicy && request.PolicyReason != "" {
			fmt.Fprintf(out, "Policy: %s\n", request.PolicyReason)
		}
		fmt.Fprintf(out, "Risk: %s\n", request.Policy.Risk)
		if request.Policy.Preview != "" && request.Policy.Preview != PreviewNone {
			fmt.Fprintf(out, "Preview: %s\n", request.Policy.Preview)
		}
		if len(request.Arguments) > 0 {
			fmt.Fprintf(out, "Arguments: %s\n", string(request.Arguments))
		}
		fmt.Fprint(out, "Approve? [y/N] ")

		answer, err := reader.ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return ApprovalDecision{}, err
		}
		answer = strings.ToLower(strings.TrimSpace(answer))
		return ApprovalDecision{Approved: answer == "y" || answer == "yes"}, nil
	}
}
