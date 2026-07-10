package tool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ToolPolicyAction controls whether a tool call may execute or needs approval.
type ToolPolicyAction string

const (
	ToolPolicyAllow           ToolPolicyAction = "allow"
	ToolPolicyRequireApproval ToolPolicyAction = "require_approval"
	ToolPolicyDeny            ToolPolicyAction = "deny"
)

// ToolPolicyRequest describes one validated tool call before approval or execution.
type ToolPolicyRequest struct {
	ToolCallID  string
	ToolName    string
	Description string
	Arguments   json.RawMessage
	Policy      ApprovalPolicy
	Round       int
	Model       string
}

// ToolPolicyDecision is returned by an application-owned policy hook.
type ToolPolicyDecision struct {
	Action ToolPolicyAction
	Reason string
}

// ToolPolicyHook evaluates every toolbox call after argument validation.
type ToolPolicyHook func(context.Context, ToolPolicyRequest) (ToolPolicyDecision, error)

// ErrToolPolicyDenied identifies a central tool-policy denial or evaluation failure.
var ErrToolPolicyDenied = errors.New("tool policy denied")

// ToolPolicyError preserves policy failure context without exposing arguments.
type ToolPolicyError struct {
	ToolName string
	Reason   string
	Err      error
}

func (err *ToolPolicyError) Error() string {
	if err == nil {
		return "<nil>"
	}
	message := fmt.Sprintf("tool %q: %v", err.ToolName, ErrToolPolicyDenied)
	if err.Reason != "" {
		message += ": " + err.Reason
	} else if err.Err != nil {
		message += ": policy evaluation failed"
	}
	return message
}

func (err *ToolPolicyError) Unwrap() error {
	if err == nil {
		return nil
	}
	return err.Err
}

func (err *ToolPolicyError) Is(target error) bool {
	return target == ErrToolPolicyDenied
}

func evaluateToolPolicy(ctx context.Context, hook ToolPolicyHook, request ToolPolicyRequest) (ToolPolicyAction, string, error) {
	action := ToolPolicyAllow
	if request.Policy.requiresApproval() {
		action = ToolPolicyRequireApproval
	}
	if hook == nil {
		return action, "", nil
	}
	decision, err := hook(ctx, request)
	if err != nil {
		return ToolPolicyDeny, "", &ToolPolicyError{ToolName: request.ToolName, Err: err}
	}
	decision.Reason = strings.TrimSpace(decision.Reason)
	switch decision.Action {
	case ToolPolicyAllow:
		return action, decision.Reason, nil
	case ToolPolicyRequireApproval:
		return ToolPolicyRequireApproval, decision.Reason, nil
	case ToolPolicyDeny:
		return ToolPolicyDeny, decision.Reason, &ToolPolicyError{ToolName: request.ToolName, Reason: decision.Reason}
	default:
		return ToolPolicyDeny, "", &ToolPolicyError{
			ToolName: request.ToolName,
			Err:      fmt.Errorf("invalid tool policy action %q", decision.Action),
		}
	}
}
