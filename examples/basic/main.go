package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
	"github.com/beowulf20/kisaragi-kit/pkg/llm/agent"
	openaiadapter "github.com/beowulf20/kisaragi-kit/pkg/llm/provider/openai"
	llmtool "github.com/beowulf20/kisaragi-kit/pkg/llm/tool"
)

type weatherInput struct {
	City string `json:"city" description:"City to check"`
}

type weatherOutput struct {
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

	tools := llmtool.NewToolbox()
	err = tools.RegisterTool(llmtool.NewTool("weather", "Gets current weather.", func(_ context.Context, input weatherInput) (weatherOutput, error) {
		return weatherOutput{Summary: "clear skies in " + input.City}, nil
	}))
	if err != nil {
		log.Fatal(err)
	}

	model := getenv("OPENAI_MODEL", "gpt-4o-mini")
	reasoningEffort, err := getenvReasoningEffort("OPENAI_REASONING_EFFORT")
	if err != nil {
		log.Fatal(err)
	}
	assistant, err := agent.NewAgent(agent.NewAgentInput{
		Name:         "assistant",
		SystemPrompt: "Answer briefly. Use tools when they help.",
		Config: llm.CompletionCallInput{
			Client:          client,
			Model:           model,
			ReasoningEffort: reasoningEffort,
			Tools:           *tools,
		},
		Hooks: agent.Hooks{
			OnContentDelta: func(delta string) { fmt.Print(delta) },
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	output, err := assistant.CallWithUserMessage("What is the weather in Curitiba?")
	if err != nil {
		log.Fatal(err)
	}
	if output.Content == "" {
		fmt.Println()
	}
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
