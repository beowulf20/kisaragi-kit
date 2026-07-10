package guardrail

import (
	"context"
	"strings"
	"testing"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

func TestToolMetadataLeakGuardrailCollectsToolAndNestedParameterNames(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{
		Name: "internal_lookup",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"short": map[string]any{"type": "string"},
				"request_context": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"customer_reference": map[string]any{"type": "string"},
					},
				},
			},
		},
	}}, ToolMetadataLeakConfig{})

	for _, value := range []string{"internal_lookup", "request_context", "customer_reference"} {
		assertToolMetadataDecision(t, guardrail, assistantInput(value), llm.MessageGuardrailBlock)
	}
	assertToolMetadataDecision(t, guardrail, assistantInput("short"), llm.MessageGuardrailAllow)
}

func TestToolMetadataLeakGuardrailUsesConservativeParameterDefaultAndIgnoresDescriptions(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{
		Name:        "internal_fn",
		Description: "secret_description",
		Parameters: map[string]any{
			"properties": map[string]any{
				"abcdefgh": map[string]any{"type": "string"},
				"abcdefg":  map[string]any{"type": "string"},
			},
		},
	}}, ToolMetadataLeakConfig{})

	assertToolMetadataDecision(t, guardrail, assistantInput("abcdefgh"), llm.MessageGuardrailBlock)
	assertToolMetadataDecision(t, guardrail, assistantInput("abcdefg secret_description"), llm.MessageGuardrailAllow)
}

func TestToolMetadataLeakGuardrailAppliesExclusionsAndAdditionalNames(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{
		Name: "internal_lookup",
		Parameters: map[string]any{
			"properties": map[string]any{"customer_reference": map[string]any{"type": "string"}},
		},
	}}, ToolMetadataLeakConfig{
		ExcludedNames:          []string{"INTERNAL_LOOKUP", "customer_reference"},
		AdditionalBlockedNames: []string{"ops"},
	})

	assertToolMetadataDecision(t, guardrail, assistantInput("internal_lookup customer_reference"), llm.MessageGuardrailAllow)
	assertToolMetadataDecision(t, guardrail, assistantInput("OPS"), llm.MessageGuardrailBlock)
}

func TestToolMetadataLeakGuardrailMatchesIdentifierBoundaries(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{Name: "internal_lookup"}}, ToolMetadataLeakConfig{})

	assertToolMetadataDecision(t, guardrail, assistantInput("Call `INTERNAL_LOOKUP` now."), llm.MessageGuardrailBlock)
	assertToolMetadataDecision(t, guardrail, assistantInput("preinternal_lookupx"), llm.MessageGuardrailAllow)
}

func TestToolMetadataLeakGuardrailChecksAssistantContentAndReasoningPhases(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{Name: "internal_lookup"}}, ToolMetadataLeakConfig{})
	for _, phase := range []llm.MessageGuardrailPhase{
		llm.MessageGuardrailPhaseAssistantContentDelta,
		llm.MessageGuardrailPhaseAssistantReasoningDelta,
		llm.MessageGuardrailPhaseAssistantReasoningFinal,
		llm.MessageGuardrailPhaseAssistantFinal,
	} {
		assertToolMetadataDecision(t, guardrail, llm.MessageGuardrailInput{
			Phase:   phase,
			Message: llm.NewAssistantMessage("internal_lookup"),
		}, llm.MessageGuardrailBlock)
	}
}

func TestToolMetadataLeakGuardrailAllowsStructuralNamesAndKeys(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{
		Name: "internal_lookup",
		Parameters: map[string]any{
			"properties": map[string]any{"customer_reference": map[string]any{"type": "string"}},
		},
	}}, ToolMetadataLeakConfig{})
	input := llm.MessageGuardrailInput{
		Phase: llm.MessageGuardrailPhaseAssistantFinal,
		Message: llm.NewAssistantToolCallMessage("", []llm.ToolCall{{
			Name:      "internal_lookup",
			Arguments: `{"customer_reference":"safe value"}`,
		}}),
	}
	assertToolMetadataDecision(t, guardrail, input, llm.MessageGuardrailAllow)
}

func TestToolMetadataLeakGuardrailBlocksToolArgumentStringValues(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{Name: "internal_lookup"}}, ToolMetadataLeakConfig{})
	input := llm.MessageGuardrailInput{
		Phase: llm.MessageGuardrailPhaseAssistantFinal,
		Message: llm.NewAssistantToolCallMessage("", []llm.ToolCall{{
			Name:      "send_message",
			Arguments: `{"payload":{"items":["safe","internal_lookup"]}}`,
		}}),
	}
	assertToolMetadataDecision(t, guardrail, input, llm.MessageGuardrailBlock)
}

func TestToolMetadataLeakGuardrailIgnoresMalformedToolArguments(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{Name: "internal_lookup"}}, ToolMetadataLeakConfig{})
	input := llm.MessageGuardrailInput{
		Phase: llm.MessageGuardrailPhaseAssistantFinal,
		Message: llm.NewAssistantToolCallMessage("", []llm.ToolCall{{
			Name:      "send_message",
			Arguments: `{"value":"internal_lookup"`,
		}}),
	}
	assertToolMetadataDecision(t, guardrail, input, llm.MessageGuardrailAllow)
}

func TestToolMetadataLeakGuardrailChecksToolResultValuesNotKeys(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{Name: "internal_lookup"}}, ToolMetadataLeakConfig{})

	assertToolMetadataDecision(t, guardrail, llm.MessageGuardrailInput{
		Phase:   llm.MessageGuardrailPhaseToolResult,
		Message: llm.NewToolMessage("call_1", `{"internal_lookup":"safe"}`),
	}, llm.MessageGuardrailAllow)
	assertToolMetadataDecision(t, guardrail, llm.MessageGuardrailInput{
		Phase:   llm.MessageGuardrailPhaseToolResult,
		Message: llm.NewToolMessage("call_1", `{"value":"internal_lookup"}`),
	}, llm.MessageGuardrailBlock)
	assertToolMetadataDecision(t, guardrail, llm.MessageGuardrailInput{
		Phase:   llm.MessageGuardrailPhaseToolResult,
		Message: llm.NewToolMessage("call_1", "internal_lookup failed"),
	}, llm.MessageGuardrailBlock)
}

func TestToolMetadataLeakGuardrailAppliesOnlyToLeakBearingInputRoles(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{Name: "internal_lookup"}}, ToolMetadataLeakConfig{})

	for _, message := range []llm.Message{
		llm.NewSystemMessage("internal_lookup"),
		llm.NewUserMessage("internal_lookup"),
	} {
		assertToolMetadataDecision(t, guardrail, llm.MessageGuardrailInput{
			Phase:   llm.MessageGuardrailPhaseInput,
			Message: message,
		}, llm.MessageGuardrailAllow)
	}
	for _, message := range []llm.Message{
		llm.NewAssistantMessage("internal_lookup"),
		llm.NewToolMessage("call_1", "internal_lookup"),
	} {
		assertToolMetadataDecision(t, guardrail, llm.MessageGuardrailInput{
			Phase:   llm.MessageGuardrailPhaseInput,
			Message: message,
		}, llm.MessageGuardrailBlock)
	}
	assertToolMetadataDecision(t, guardrail, llm.MessageGuardrailInput{
		Phase:   llm.MessageGuardrailPhaseApprovalDecision,
		Message: llm.NewUserMessage(`{"tool":"internal_lookup"}`),
	}, llm.MessageGuardrailAllow)
}

func TestToolMetadataLeakGuardrailReasonDoesNotExposeMatchedName(t *testing.T) {
	guardrail := newTestToolMetadataGuardrail(t, []llmtool.ChatTool{{Name: "internal_lookup"}}, ToolMetadataLeakConfig{})
	decision, err := guardrail.CheckMessage(context.Background(), assistantInput("internal_lookup"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.ToLower(decision.Reason), "internal_lookup") {
		t.Fatalf("reason exposed blocked name: %q", decision.Reason)
	}
}

func TestToolMetadataLeakGuardrailRejectsInvalidOrEmptyConfig(t *testing.T) {
	if _, err := NewToolMetadataLeakGuardrail(nil, ToolMetadataLeakConfig{MinParameterNameLength: -1}); err == nil {
		t.Fatal("expected invalid minimum length error")
	}
	if _, err := NewToolMetadataLeakGuardrail(nil, ToolMetadataLeakConfig{}); err == nil {
		t.Fatal("expected empty blocklist error")
	}
	if _, err := NewToolMetadataLeakGuardrail(nil, ToolMetadataLeakConfig{AdditionalBlockedNames: []string{" "}}); err == nil {
		t.Fatal("expected empty additional name error")
	}
}

func assistantInput(content string) llm.MessageGuardrailInput {
	return llm.MessageGuardrailInput{
		Phase:   llm.MessageGuardrailPhaseAssistantContentDelta,
		Message: llm.NewAssistantMessage(content),
	}
}

func newTestToolMetadataGuardrail(t *testing.T, tools []llmtool.ChatTool, config ToolMetadataLeakConfig) llm.MessageGuardrail {
	t.Helper()
	guardrail, err := NewToolMetadataLeakGuardrail(tools, config)
	if err != nil {
		t.Fatal(err)
	}
	return guardrail
}

func assertToolMetadataDecision(t *testing.T, guardrail llm.MessageGuardrail, input llm.MessageGuardrailInput, want llm.MessageGuardrailAction) {
	t.Helper()
	decision, err := guardrail.CheckMessage(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Action != want {
		t.Fatalf("action = %q, want %q for %#v", decision.Action, want, input)
	}
}
