package llm

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/require"
)

func Test_ChatGPT4(t *testing.T) {
	client := openai.NewClient(os.Getenv("OPENAI_TOKEN"))

	evaluators := map[string]Evaluator{
		"gpt3p5":  NewChatGPT3p5(client),
		"gpt4":    NewChatGPT4(client),
		"llama7b": NewLlama("./models/7B/ggml-model-f32.bin"),
	}
	service := NewService(
		"From now on, all answers should be in capital letters.",
		evaluators,
	)

	fmt.Println("models loaded")

	ctx := context.Background()
	prompt := "What is the answer to life, the universe and everything and " +
		"where is the question from?"

	t.Run("llama7b", func(t *testing.T) {
		resp, err := service.Evaluate(
			ctx,
			"llama7b",
			prompt,
		)
		require.NoError(t, err)
		fmt.Println("> llama7b:", resp)
	})

	t.Run("gpt3p5", func(t *testing.T) {
		resp, err := service.Evaluate(
			ctx,
			"gpt3p5",
			prompt,
		)
		require.NoError(t, err)
		fmt.Println("> gpt3p5:", resp)
	})

	t.Run("gpt4", func(t *testing.T) {
		resp, err := service.Evaluate(
			ctx,
			"gpt4",
			prompt,
		)
		require.NoError(t, err)
		fmt.Println("> gpt4:", resp)
	})
}
