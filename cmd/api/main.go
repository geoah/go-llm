package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/go-chi/chi/v5"
	"github.com/sashabaranov/go-openai"

	"github.com/geoah/go-llm"
)

type EvaluateRequest struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`
}

type EvaluateResponse struct {
	Result string `json:"result"`
}

func main() {
	client := openai.NewClient(os.Getenv("OPENAI_TOKEN"))

	evaluators := map[string]llm.Evaluator{
		"gpt3p5": llm.NewChatGPT3p5(client),
		"gpt4":   llm.NewChatGPT4(client),
		// "llama7b": llm.NewLlama("./models/7B/ggml-model-f32.bin"),
	}

	service := llm.NewService(
		os.Getenv("PREFIX"),
		evaluators,
	)

	r := chi.NewRouter()

	handleEvaluate := func(w http.ResponseWriter, r *http.Request) {
		var req EvaluateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		result, err := service.Evaluate(r.Context(), req.Model, req.Prompt)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		res := EvaluateResponse{Result: result}
		if err := json.NewEncoder(w).Encode(res); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	r.Route("/api/v1", func(r chi.Router) {
		r.Post("/evaluate", handleEvaluate)
	})

	log.Println("Starting server on :8080...")
	if err := http.ListenAndServe(":8080", r); err != nil {
		log.Fatal(err)
	}
}
