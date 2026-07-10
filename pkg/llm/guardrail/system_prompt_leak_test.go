package guardrail

import (
	"context"
	"strings"
	"testing"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

func TestSystemPromptLeakGuardrailBlocksCoveredPrompt(t *testing.T) {
	guardrail, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{
		Threshold:     0.4,
		MinMatchWords: 3,
	})
	if err != nil {
		t.Fatal(err)
	}

	decision, err := guardrail.CheckMessage(context.Background(), llm.MessageGuardrailInput{
		Message: llm.NewAssistantMessage("ALPHA, beta gamma delta! unrelated"),
		Messages: []llm.Message{
			llm.NewSystemMessage("alpha beta gamma delta epsilon zeta eta theta"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != llm.MessageGuardrailBlock {
		t.Fatalf("action = %q, want block", decision.Action)
	}
	if strings.Contains(decision.Reason, "alpha beta") {
		t.Fatalf("reason leaked matched prompt text: %q", decision.Reason)
	}
}

func TestSystemPromptLeakGuardrailAllowsSmallMatch(t *testing.T) {
	guardrail, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{
		Threshold:     0.75,
		MinMatchWords: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := guardrail.CheckMessage(context.Background(), llm.MessageGuardrailInput{
		Message:  llm.NewAssistantMessage("alpha beta gamma"),
		Messages: []llm.Message{llm.NewSystemMessage("alpha beta gamma delta epsilon zeta eta theta")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != llm.MessageGuardrailAllow {
		t.Fatalf("action = %q, want allow", decision.Action)
	}
}

func TestSystemPromptLeakGuardrailCombinesDisjointCoverageAtBoundary(t *testing.T) {
	guardrail, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{
		Threshold:     0.4,
		MinMatchWords: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := guardrail.CheckMessage(context.Background(), llm.MessageGuardrailInput{
		Message:  llm.NewAssistantMessage("alpha beta unrelated words iota kappa"),
		Messages: []llm.Message{llm.NewSystemMessage("alpha beta gamma delta epsilon zeta eta theta iota kappa")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != llm.MessageGuardrailBlock {
		t.Fatalf("action = %q, want boundary block", decision.Action)
	}
}

func TestSystemPromptLeakGuardrailChecksEachSystemMessageIndependently(t *testing.T) {
	guardrail, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{
		Threshold:     1,
		MinMatchWords: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := guardrail.CheckMessage(context.Background(), llm.MessageGuardrailInput{
		Message: llm.NewAssistantMessage("small secret prompt"),
		Messages: []llm.Message{
			llm.NewSystemMessage("one two three four five six seven eight nine ten eleven twelve"),
			llm.NewSystemMessage("small secret prompt"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != llm.MessageGuardrailBlock {
		t.Fatalf("action = %q, want short second prompt block", decision.Action)
	}
}

func TestSystemPromptLeakGuardrailUsesDefaults(t *testing.T) {
	guardrail, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := guardrail.CheckMessage(context.Background(), llm.MessageGuardrailInput{
		Message: llm.NewAssistantMessage("one two three four five six seven eight"),
		Messages: []llm.Message{llm.NewSystemMessage(
			"one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour twentyfive twentysix twentyseven twentyeight twentynine thirty thirtyone thirtytwo",
		)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != llm.MessageGuardrailBlock {
		t.Fatalf("action = %q, want default 25 percent coverage block", decision.Action)
	}
}

func TestSystemPromptLeakGuardrailChecksToolArguments(t *testing.T) {
	guardrail, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{
		Threshold:     1,
		MinMatchWords: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := guardrail.CheckMessage(context.Background(), llm.MessageGuardrailInput{
		Message: llm.NewAssistantToolCallMessage("", []llm.ToolCall{{
			Name:      "send",
			Arguments: `{"text":"secret launch code blue"}`,
		}}),
		Messages: []llm.Message{llm.NewSystemMessage("secret launch code blue")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != llm.MessageGuardrailBlock {
		t.Fatalf("action = %q, want block", decision.Action)
	}
}

func TestSystemPromptLeakGuardrailRequiresFullShortPrompt(t *testing.T) {
	guardrail, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{
		Threshold:     0.2,
		MinMatchWords: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	input := llm.MessageGuardrailInput{
		Message:  llm.NewAssistantMessage("do not reveal this"),
		Messages: []llm.Message{llm.NewSystemMessage("do not reveal this")},
	}
	decision, err := guardrail.CheckMessage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != llm.MessageGuardrailBlock {
		t.Fatalf("action = %q, want block", decision.Action)
	}
}

func TestSystemPromptLeakGuardrailIgnoresNonAssistantMessages(t *testing.T) {
	guardrail, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := guardrail.CheckMessage(context.Background(), llm.MessageGuardrailInput{
		Message:  llm.NewUserMessage("secret launch code blue"),
		Messages: []llm.Message{llm.NewSystemMessage("secret launch code blue")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != llm.MessageGuardrailAllow {
		t.Fatalf("action = %q, want allow", decision.Action)
	}
}

func TestSystemPromptLeakGuardrailRejectsInvalidConfig(t *testing.T) {
	if _, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{Threshold: 1.1}); err == nil {
		t.Fatal("expected invalid threshold error")
	}
	if _, err := NewSystemPromptLeakGuardrail(SystemPromptLeakConfig{MinMatchWords: -1}); err == nil {
		t.Fatal("expected invalid minimum match words error")
	}
}
