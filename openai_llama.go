package llm

import (
	"container/ring"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/gotzmann/llama.go/pkg/llama"
	"github.com/gotzmann/llama.go/pkg/ml"
)

type Llama struct {
	lctx   *llama.Context
	params llama.ModelParams
}

func NewLlama(model string) *Llama {
	lctx, err := llama.LoadModel(
		model,
		false, // TODO: make silent
	)
	if err != nil {
		panic(fmt.Errorf("failed to load model: %w", err))
	}

	fmt.Println("model loaded:", model)

	params := llama.ModelParams{
		Model: model,

		MaxThreads: 4, // TODO: make configurable

		UseAVX:  true, // TODO: make configurable
		UseNEON: true, // TODO: make configurable

		Interactive: false, // TODO: make configurable

		CtxSize:      128,
		Seed:         -1,
		PredictCount: 128,
		RepeatLastN:  0,
		PartsCount:   -1,
		BatchSize:    8,

		TopK:          40,
		TopP:          0.95,
		Temp:          0.8,
		RepeatPenalty: 1.10,

		MemoryFP16: true,
	}

	return &Llama{
		lctx:   lctx,
		params: params,
	}
}

func (o *Llama) Evaluate(ctx context.Context, prefix, prompt string) (string, error) {
	// tokenize the prompt
	prompt = strings.Join([]string{prefix, prompt}, "\n")
	embdInp := ml.Tokenize(o.lctx.Vocab, prompt, true)
	tokenNewline := ml.Tokenize(o.lctx.Vocab, "\n", false)[0]

	var embd []uint32

	// Initialize the ring buffer
	lastNTokens := ring.New(int(o.params.CtxSize))

	for i := 0; i < int(o.params.CtxSize); i++ {
		lastNTokens.Value = uint32(0)
		lastNTokens = lastNTokens.Next()
	}

	// A function to append a token to the ring buffer
	appendToken := func(token uint32) {
		lastNTokens.Value = token
		lastNTokens = lastNTokens.Next()
	}

	// inputNoEcho := false
	pastCount := uint32(0)
	remainCount := o.params.PredictCount
	consumedCount := uint32(0)
	// evalPerformance := make([]int64, 0, o.params.PredictCount)

	final := ""

	for remainCount != 0 || o.params.Interactive {

		fmt.Println("remainCount:", remainCount)

		// --- predict

		if len(embd) > 0 {

			fmt.Println("embd:", len(embd))

			// infinite text generation via context swapping
			// if we run out of context:
			// - take the n_keep first tokens from the original prompt (via n_past)
			// - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch

			if pastCount+uint32(len(embd)) > o.params.CtxSize {
				leftCount := pastCount - o.params.KeepCount
				pastCount = o.params.KeepCount

				// insert n_left/2 tokens at the start of embd from last_n_tokens
				// embd = append(lastNTokens[:leftCount/2], embd...)
				embd = append(llama.ExtractTokens(lastNTokens.Move(-int(leftCount/2)), int(leftCount/2)), embd...)
			}

			// start := time.Now().UnixNano()
			if err := llama.Eval(o.lctx, embd, uint32(len(embd)), pastCount, o.params); err != nil {
				fmt.Printf("\n[ERROR] Failed to eval")
				os.Exit(1)
			}
			// evalPerformance = append(evalPerformance, time.Now().UnixNano()-start)
		}

		pastCount += uint32(len(embd))
		embd = []uint32{}

		if len(embdInp) <= int(consumedCount) { // && !isInteracting {
			fmt.Println(">>", len(embdInp), consumedCount)

			if o.params.IgnoreEOS {
				o.lctx.Logits[ml.TOKEN_EOS] = 0
			}

			/*
				id := llama.SampleTopPTopK(ctx,
					lastNTokens[params.ctxSize-params.repeatLastN:], params.repeatLastN,
					params.topK, params.topP, params.temp, params.repeatPenalty)

				lastNTokens = lastNTokens[1:] ////last_n_tokens.erase(last_n_tokens.begin());
				lastNTokens = append(lastNTokens, id)

			*/
			id := llama.SampleTopPTopK(o.lctx,
				lastNTokens, o.params.RepeatLastN,
				o.params.TopK, o.params.TopP, o.params.Temp, o.params.RepeatPenalty)

			appendToken(id)

			// replace end of text token with newline token when in interactive mode
			if id == ml.TOKEN_EOS && o.params.Interactive && !o.params.Instruct {
				id = tokenNewline
			}

			// add it to the context
			embd = append(embd, id)

			// echo this to console
			// inputNoEcho = false

			// decrement remaining sampling budget
			remainCount--

		} else {

			fmt.Println("else")
			// some user input remains from prompt or interaction, forward it to processing
			/*
				for len(embdInp) > int(consumedCount) {
					embd = append(embd, embdInp[consumedCount])
					if len(lastNTokens) > 0 {
						lastNTokens = lastNTokens[1:]
					}
					lastNTokens = append(lastNTokens, embdInp[consumedCount])
					consumedCount++
					if len(embd) >= int(params.batchSize) {
						break
					}
				}
			*/
			for len(embdInp) > int(consumedCount) {
				embd = append(embd, embdInp[consumedCount])
				appendToken(embdInp[consumedCount])
				consumedCount++
				if len(embd) >= int(o.params.BatchSize) {
					break
				}
			}
		}

		// --- display text

		for _, id := range embd {

			token := ml.Token2Str(o.lctx.Vocab, id)
			final += token

			if len(strings.TrimSpace(final)) < len(strings.TrimSpace(prompt)) {
				continue
			}

			out := strings.Split(final, prompt)

			if len(out) == 2 && token == "\n" {
				continue
			}

			// if len(strings.TrimSpace(final)) == len(strings.TrimSpace(prompt)) && (token != "\n") && (len(out) == 2) {
			// 	Colorize("\n\n[magenta]▒▒▒ [light_yellow]" + strings.TrimSpace(prompt) + "\n[light_blue]▒▒▒ ")
			// 	continue
			// }

			// Colorize("[white]" + token)
		}
	}

	return final, nil
}
