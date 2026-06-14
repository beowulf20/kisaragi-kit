package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/agent"
	openaiadapter "github.com/beowulf20/kisaragi-kit/pkg/llm/provider/openai"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

type noteInput struct {
	Title string `json:"title" description:"Short note title"`
	Body  string `json:"body" description:"Note body to save"`
}

type noteOutput struct {
	ID      string `json:"id"`
	Summary string `json:"summary"`
}

func main() {
	client, _, err := openaiadapter.NewClient(openaiadapter.ClientConfig{
		BaseURL: getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		APIKey:  os.Getenv("OPENAI_API_KEY"),
		Timeout: 60 * time.Second,
	})
	if err != nil {
		log.Fatal(err)
	}

	tools := llmtool.NewToolbox(llmtool.WithApprovalHook(promptApproval))
	err = tools.RegisterTool(llmtool.NewTool("save_note", "Saves a note after human approval.", func(_ context.Context, input noteInput) (noteOutput, error) {
		return noteOutput{
			ID:      "note_demo_001",
			Summary: input.Title + ": " + input.Body,
		}, nil
	}, llmtool.WithApproval(llmtool.ApprovalPolicy{
		Mode:        llmtool.ApprovalAlways,
		Risk:        llmtool.RiskHigh,
		Preview:     llmtool.PreviewPayload,
		Description: "Persist a user-supplied note.",
	})))
	if err != nil {
		log.Fatal(err)
	}

	model := getenv("OPENAI_MODEL", "gpt-4o-mini")
	reasoningEffort, err := getenvReasoningEffort("OPENAI_REASONING_EFFORT")
	if err != nil {
		log.Fatal(err)
	}
	assistant, err := agent.NewAgent(agent.NewAgentInput{
		Name:         "approval-demo",
		SystemPrompt: "Save the requested note using tools. Keep the final response brief.",
		Config: llm.CompletionCallInput{
			Client:          client,
			Model:           model,
			ReasoningEffort: reasoningEffort,
			Tools:           *tools,
			ApprovalDecisionMessages: llm.ApprovalDecisionMessages{
				AppendAccepted: true,
				AppendRejected: true,
			},
		},
		Hooks: agent.Hooks{
			OnContentDelta: func(delta string) { fmt.Print(delta) },
			OnToolCall: func(call llm.ToolCall) {
				fmt.Printf("\nTool requested: %s %s\n", call.Name, call.Arguments)
			},
			OnToolResult: func(call llm.ToolCall) {
				fmt.Printf("\nTool finished: %s\n", call.Name)
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	output, err := assistant.CallWithUserMessage("Save a note titled Launch checklist with body Confirm approvals before deploy.")
	if err != nil {
		log.Fatal(err)
	}
	if output.Content == "" {
		fmt.Println()
	}
}

func promptApproval(_ context.Context, request llmtool.ApprovalRequest) (llmtool.ApprovalDecision, error) {
	fmt.Println()
	fmt.Println("Tool approval required")
	fmt.Printf("Tool: %s\n", request.ToolName)
	fmt.Printf("Mode: %s\n", request.Policy.Mode)
	fmt.Printf("Risk: %s\n", request.Policy.Risk)
	fmt.Printf("Preview: %s\n", request.Policy.Preview)
	if request.Policy.Description != "" {
		fmt.Printf("Intent: %s\n", request.Policy.Description)
	}
	fmt.Printf("Arguments: %s\n", request.Arguments)
	fmt.Print("Approve? [y/N] ")

	var answer string
	if _, err := fmt.Scanln(&answer); err != nil {
		return llmtool.ApprovalDecision{Approved: false, Reason: "no approval entered"}, nil
	}
	answer = strings.ToLower(strings.TrimSpace(answer))
	if answer == "y" || answer == "yes" {
		return llmtool.ApprovalDecision{Approved: true, Reason: "approved in terminal"}, nil
	}
	return llmtool.ApprovalDecision{Approved: false, Reason: "rejected in terminal"}, nil
}

func getenv(key, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	return value
}

func getenvReasoningEffort(key string) (llm.ReasoningEffort, error) {
	switch value := os.Getenv(key); value {
	case "":
		return "", nil
	case string(llm.ReasoningEffortNone):
		return llm.ReasoningEffortNone, nil
	case string(llm.ReasoningEffortMinimal):
		return llm.ReasoningEffortMinimal, nil
	case string(llm.ReasoningEffortLow):
		return llm.ReasoningEffortLow, nil
	case string(llm.ReasoningEffortMedium):
		return llm.ReasoningEffortMedium, nil
	case string(llm.ReasoningEffortHigh):
		return llm.ReasoningEffortHigh, nil
	case string(llm.ReasoningEffortXHigh):
		return llm.ReasoningEffortXHigh, nil
	default:
		return "", fmt.Errorf("%s must be one of none, minimal, low, medium, high, xhigh", key)
	}
}
