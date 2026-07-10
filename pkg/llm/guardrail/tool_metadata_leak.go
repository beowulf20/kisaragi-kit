package guardrail

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

const DefaultToolMetadataLeakMinParameterNameLength = 8

// ToolMetadataLeakConfig controls which internal tool identifiers are blocked.
type ToolMetadataLeakConfig struct {
	MinParameterNameLength int
	ExcludedNames          []string
	AdditionalBlockedNames []string
}

type toolMetadataLeakGuardrail struct {
	blockedNames []string
}

// NewToolMetadataLeakGuardrail builds a snapshot of internal names from tools.
func NewToolMetadataLeakGuardrail(tools []llmtool.ChatTool, config ToolMetadataLeakConfig) (llm.MessageGuardrail, error) {
	if config.MinParameterNameLength < 0 {
		return nil, fmt.Errorf("tool metadata leak minimum parameter name length cannot be negative")
	}
	if config.MinParameterNameLength == 0 {
		config.MinParameterNameLength = DefaultToolMetadataLeakMinParameterNameLength
	}

	excluded := make(map[string]struct{}, len(config.ExcludedNames))
	for _, name := range config.ExcludedNames {
		if normalized := normalizeIdentifier(name); normalized != "" {
			excluded[normalized] = struct{}{}
		}
	}
	blocked := make(map[string]struct{})
	addAutomatic := func(name string) {
		normalized := normalizeIdentifier(name)
		if normalized == "" {
			return
		}
		if _, skip := excluded[normalized]; !skip {
			blocked[normalized] = struct{}{}
		}
	}
	for _, tool := range tools {
		addAutomatic(tool.Name)
		collectParameterNames(tool.Parameters, config.MinParameterNameLength, addAutomatic)
	}
	for index, name := range config.AdditionalBlockedNames {
		normalized := normalizeIdentifier(name)
		if normalized == "" {
			return nil, fmt.Errorf("additional blocked name %d cannot be empty", index)
		}
		blocked[normalized] = struct{}{}
	}
	if len(blocked) == 0 {
		return nil, fmt.Errorf("tool metadata leak guardrail has no blocked names")
	}

	blockedNames := make([]string, 0, len(blocked))
	for name := range blocked {
		blockedNames = append(blockedNames, name)
	}
	sort.Strings(blockedNames)
	return toolMetadataLeakGuardrail{blockedNames: blockedNames}, nil
}

func (toolMetadataLeakGuardrail) Name() string {
	return "tool_metadata_leak"
}

func (guardrail toolMetadataLeakGuardrail) CheckMessage(_ context.Context, input llm.MessageGuardrailInput) (llm.MessageGuardrailDecision, error) {
	for _, candidate := range toolMetadataCandidates(input) {
		for _, blockedName := range guardrail.blockedNames {
			if containsIdentifier(candidate.text, blockedName) {
				return llm.MessageGuardrailDecision{
					Action: llm.MessageGuardrailBlock,
					Reason: "internal tool metadata detected in " + candidate.source,
				}, nil
			}
		}
	}
	return llm.MessageGuardrailDecision{Action: llm.MessageGuardrailAllow}, nil
}

type toolMetadataCandidate struct {
	source string
	text   string
}

func toolMetadataCandidates(input llm.MessageGuardrailInput) []toolMetadataCandidate {
	switch input.Phase {
	case llm.MessageGuardrailPhaseApprovalDecision:
		return nil
	case llm.MessageGuardrailPhaseInput:
		switch input.Message.Type {
		case llm.Assistant:
			return assistantMetadataCandidates(input.Message, "assistant history")
		case llm.Tool:
			return contentValueCandidates(input.Message.Content, "tool history")
		default:
			return nil
		}
	case llm.MessageGuardrailPhaseAssistantContentDelta:
		return []toolMetadataCandidate{{source: "assistant content", text: input.Message.Content}}
	case llm.MessageGuardrailPhaseAssistantReasoningDelta, llm.MessageGuardrailPhaseAssistantReasoningFinal:
		return []toolMetadataCandidate{{source: "assistant reasoning", text: input.Message.Content}}
	case llm.MessageGuardrailPhaseAssistantFinal:
		return assistantMetadataCandidates(input.Message, "assistant response")
	case llm.MessageGuardrailPhaseToolResult:
		return contentValueCandidates(input.Message.Content, "tool result")
	default:
		return nil
	}
}

func assistantMetadataCandidates(message llm.Message, source string) []toolMetadataCandidate {
	candidates := []toolMetadataCandidate{{source: source, text: message.Content}}
	for _, call := range message.ToolCalls {
		values, validJSON := jsonStringValues(call.Arguments)
		if !validJSON {
			continue
		}
		for _, value := range values {
			candidates = append(candidates, toolMetadataCandidate{source: "tool argument value", text: value})
		}
	}
	return candidates
}

func contentValueCandidates(content string, source string) []toolMetadataCandidate {
	values, validJSON := jsonStringValues(content)
	if !validJSON {
		return []toolMetadataCandidate{{source: source, text: content}}
	}
	candidates := make([]toolMetadataCandidate, 0, len(values))
	for _, value := range values {
		candidates = append(candidates, toolMetadataCandidate{source: source, text: value})
	}
	return candidates
}

func jsonStringValues(value string) ([]string, bool) {
	var decoded any
	if err := json.Unmarshal([]byte(value), &decoded); err != nil {
		return nil, false
	}
	values := make([]string, 0)
	collectJSONStringValues(decoded, &values)
	return values, true
}

func collectJSONStringValues(value any, values *[]string) {
	switch typed := value.(type) {
	case string:
		*values = append(*values, typed)
	case []any:
		for _, item := range typed {
			collectJSONStringValues(item, values)
		}
	case map[string]any:
		for _, item := range typed {
			collectJSONStringValues(item, values)
		}
	}
}

func collectParameterNames(schema map[string]any, minLength int, add func(string)) {
	if schema == nil {
		return
	}
	if properties, ok := schema["properties"].(map[string]any); ok {
		for name, value := range properties {
			if utf8.RuneCountInString(strings.TrimSpace(name)) >= minLength {
				add(name)
			}
			if child, ok := value.(map[string]any); ok {
				collectParameterNames(child, minLength, add)
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		collectParameterNames(items, minLength, add)
	}
	if additional, ok := schema["additionalProperties"].(map[string]any); ok {
		collectParameterNames(additional, minLength, add)
	}
}

func normalizeIdentifier(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func containsIdentifier(value string, identifier string) bool {
	value = strings.ToLower(value)
	for searchFrom := 0; searchFrom < len(value); {
		relative := strings.Index(value[searchFrom:], identifier)
		if relative < 0 {
			return false
		}
		start := searchFrom + relative
		end := start + len(identifier)
		if identifierBoundaryBefore(value, start) && identifierBoundaryAfter(value, end) {
			return true
		}
		searchFrom = start + 1
	}
	return false
}

func identifierBoundaryBefore(value string, index int) bool {
	if index == 0 {
		return true
	}
	r, _ := utf8.DecodeLastRuneInString(value[:index])
	return !isIdentifierRune(r)
}

func identifierBoundaryAfter(value string, index int) bool {
	if index == len(value) {
		return true
	}
	r, _ := utf8.DecodeRuneInString(value[index:])
	return !isIdentifierRune(r)
}

func isIdentifierRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}
