package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// MessageGuardrailPhase identifies where a candidate message enters the completion flow.
type MessageGuardrailPhase string

const (
	MessageGuardrailPhaseInput                   MessageGuardrailPhase = "input"
	MessageGuardrailPhaseAssistantContentDelta   MessageGuardrailPhase = "assistant_content_delta"
	MessageGuardrailPhaseAssistantReasoningDelta MessageGuardrailPhase = "assistant_reasoning_delta"
	MessageGuardrailPhaseAssistantReasoningFinal MessageGuardrailPhase = "assistant_reasoning_final"
	MessageGuardrailPhaseAssistantFinal          MessageGuardrailPhase = "assistant_final"
	MessageGuardrailPhaseToolResult              MessageGuardrailPhase = "tool_result"
	MessageGuardrailPhaseApprovalDecision        MessageGuardrailPhase = "approval_decision"
)

// MessageGuardrailAction controls whether a candidate message may continue.
type MessageGuardrailAction string

const (
	MessageGuardrailAllow MessageGuardrailAction = "allow"
	MessageGuardrailBlock MessageGuardrailAction = "block"
)

// MessageGuardrailInput contains one candidate plus immutable completion context.
type MessageGuardrailInput struct {
	Message  Message
	Messages []Message
	Phase    MessageGuardrailPhase
	Model    string
	Round    int
	Attempt  int
}

// MessageGuardrailDecision is the allow/block result of one guardrail.
type MessageGuardrailDecision struct {
	Action MessageGuardrailAction
	Reason string
}

// MessageGuardrail inspects candidate messages before KKit advances them.
type MessageGuardrail interface {
	Name() string
	CheckMessage(context.Context, MessageGuardrailInput) (MessageGuardrailDecision, error)
}

// MessageGuardrailFunc is an adapter for ordinary message guardrail functions.
type MessageGuardrailFunc func(context.Context, MessageGuardrailInput) (MessageGuardrailDecision, error)

type namedMessageGuardrail struct {
	name  string
	check MessageGuardrailFunc
}

// NewMessageGuardrail returns a named guardrail backed by check.
func NewMessageGuardrail(name string, check MessageGuardrailFunc) MessageGuardrail {
	return namedMessageGuardrail{name: strings.TrimSpace(name), check: check}
}

func (guardrail namedMessageGuardrail) Name() string {
	return guardrail.name
}

func (guardrail namedMessageGuardrail) CheckMessage(ctx context.Context, input MessageGuardrailInput) (MessageGuardrailDecision, error) {
	if guardrail.check == nil {
		return MessageGuardrailDecision{}, errors.New("message guardrail check cannot be nil")
	}
	return guardrail.check(ctx, input)
}

// ErrMessageGuardrailBlocked identifies a guardrail block or evaluation failure.
var ErrMessageGuardrailBlocked = errors.New("message guardrail blocked")

// MessageGuardrailError reports which guardrail stopped a candidate without exposing its content.
type MessageGuardrailError struct {
	Guardrail string
	Index     int
	Phase     MessageGuardrailPhase
	Action    MessageGuardrailAction
	Reason    string
	Err       error
}

func (err *MessageGuardrailError) Error() string {
	if err == nil {
		return "<nil>"
	}
	message := fmt.Sprintf("%v at %s guardrail %q", ErrMessageGuardrailBlocked, err.Phase, err.Guardrail)
	if err.Reason != "" {
		message += ": " + err.Reason
	} else if err.Err != nil {
		message += ": guardrail evaluation failed"
	}
	return message
}

func (err *MessageGuardrailError) Unwrap() error {
	if err == nil {
		return nil
	}
	return err.Err
}

func (err *MessageGuardrailError) Is(target error) bool {
	return target == ErrMessageGuardrailBlocked
}

func validateMessageGuardrails(guardrails []MessageGuardrail) error {
	names := make(map[string]struct{}, len(guardrails))
	for index, guardrail := range guardrails {
		if guardrail == nil {
			return fmt.Errorf("message guardrail %d cannot be nil", index)
		}
		name := strings.TrimSpace(guardrail.Name())
		if name == "" {
			return fmt.Errorf("message guardrail %d name cannot be empty", index)
		}
		if _, exists := names[name]; exists {
			return fmt.Errorf("message guardrail name %q is duplicated", name)
		}
		names[name] = struct{}{}
	}
	return nil
}

func evaluateMessageGuardrails(
	ctx context.Context,
	guardrails []MessageGuardrail,
	input MessageGuardrailInput,
) error {
	if len(guardrails) == 0 {
		return nil
	}
	input.Message = cloneMessage(input.Message)
	input.Messages = cloneMessages(input.Messages)
	for index, guardrail := range guardrails {
		if err := ctx.Err(); err != nil {
			return &MessageGuardrailError{
				Guardrail: guardrail.Name(),
				Index:     index,
				Phase:     input.Phase,
				Err:       err,
			}
		}
		guardrailInput := input
		guardrailInput.Message = cloneMessage(input.Message)
		guardrailInput.Messages = cloneMessages(input.Messages)
		decision, err := guardrail.CheckMessage(ctx, guardrailInput)
		if err != nil {
			return &MessageGuardrailError{
				Guardrail: guardrail.Name(),
				Index:     index,
				Phase:     input.Phase,
				Err:       err,
			}
		}
		switch decision.Action {
		case MessageGuardrailAllow:
			continue
		case MessageGuardrailBlock:
			return &MessageGuardrailError{
				Guardrail: guardrail.Name(),
				Index:     index,
				Phase:     input.Phase,
				Action:    decision.Action,
				Reason:    strings.TrimSpace(decision.Reason),
			}
		default:
			return &MessageGuardrailError{
				Guardrail: guardrail.Name(),
				Index:     index,
				Phase:     input.Phase,
				Action:    decision.Action,
				Err:       fmt.Errorf("invalid message guardrail action %q", decision.Action),
			}
		}
	}
	return nil
}

func cloneMessage(message Message) Message {
	message.ToolCalls = append([]ToolCall(nil), message.ToolCalls...)
	return message
}

func cloneMessages(messages []Message) []Message {
	cloned := make([]Message, len(messages))
	for index, message := range messages {
		cloned[index] = cloneMessage(message)
	}
	return cloned
}
