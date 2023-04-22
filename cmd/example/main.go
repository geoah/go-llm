package main

import (
	"context"
	"fmt"

	"github.com/geoah/go-llm"
)

func main() {
	// client := openai.NewClient(os.Getenv("OPENAI_TOKEN"))

	evaluators := map[string]llm.Evaluator{
		// "gpt3p5":  NewChatGPT3p5(client),
		// "gpt4":    NewChatGPT4(client),
		"llama7b": llm.NewLlama("./models/7B/ggml-model-f32.bin"),
	}
	service := llm.NewService(
		"From now on, all answers should be in capital letters.",
		evaluators,
	)

	fmt.Println("models loaded")

	ctx := context.Background()
	prompt := "What is the answer to life, the universe and everything and " +
		"where is the question from?"

	resp, err := service.Evaluate(
		ctx,
		"llama7b",
		prompt,
	)
	if err != nil {
		panic(fmt.Errorf("llama7b: %w", err))
	}
	fmt.Println("> llama7b:", resp)
}
