package main

import (
	"context"
	"flag"
	"fmt"
	"io"
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
	printCost := flag.Bool("print-cost", false, "print the provider-reported USD cost")
	flag.Parse()

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
	} else if *printCost {
		fmt.Println()
	}
	if *printCost {
		printCostBreakdown(os.Stdout, output)
	}
}

func printCostBreakdown(w io.Writer, output *llm.CompletionCallOutput) {
	fmt.Fprintln(w, "Cost breakdown:")
	for index, event := range output.UsageEvents {
		fmt.Fprintf(w, "  Generation %d: model=%s round=%d attempt=%d prompt=%d completion=%d total=%d cost=%s\n",
			index+1,
			event.Model,
			event.Round,
			event.Attempt,
			event.Usage.PromptTokens,
			event.Usage.CompletionTokens,
			event.Usage.TotalTokens,
			formatCost(event.Usage.CostUSD),
		)
	}
	fmt.Fprintf(w, "  Total: prompt=%d completion=%d total=%d cost=%s\n",
		output.Usage.PromptTokens,
		output.Usage.CompletionTokens,
		output.Usage.TotalTokens,
		formatCost(output.Usage.CostUSD),
	)
}

func formatCost(cost *float64) string {
	if cost == nil {
		return "unavailable"
	}
	return fmt.Sprintf("$%.8f", *cost)
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
