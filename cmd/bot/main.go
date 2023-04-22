package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"syscall"

	"github.com/bwmarrin/discordgo"
	"github.com/fatih/color"
	"github.com/sashabaranov/go-openai"

	"github.com/geoah/go-llm"
)

var (
	rules = `
# Rules
1. No swearing is allowed.
2. No mention of any type of food is allowed.
3. No mention of politics is allowed.
`
	setup = `
Instructions:

You are responsible for maintaining the community guidelines of a
discord server according to rules they have set.

The rules will be provided between the lines "RULES START" and "RULES END".
The message you are evaluating is provided after the word "MESSAGE:".

If any message breaks the rules, respond with the number of the
rule that was broken. If the message breaks multiple rules, respond
with the numbers of the rules that were broken.

The response should be polite and in the following format:
"This message is against this server's community guidelines, rule #X."
It shoushould then quote the rule that was broken, prepending the rule
number with "Rule #" and prepend it with the "> " character.

If there are no rules broken or if your confidence is not over 80%%,
respond with "OK".

If you are unsure if a rule is broken, respond with "OK".

RULES START
%s
RULES END

MESSAGE:
%s
`
)

func main() {
	// Create new LLM client
	client := openai.NewClient(os.Getenv("OPENAI_TOKEN"))

	evaluators := map[string]llm.Evaluator{
		"gpt3p5": llm.NewChatGPT3p5(client),
		"gpt4":   llm.NewChatGPT4(client),
	}

	// prefix := fmt.Sprintf("%s\nRULES START\n%s\nRULES END\n", rules, setup)

	service := llm.NewService(
		"",
		evaluators,
	)

	// Create a new Discord session using the provided bot token.
	token := os.Getenv("DISCORD_TOKEN")
	dg, err := discordgo.New("Bot " + token)
	if err != nil {
		fmt.Println("error creating Discord session,", err)
		return
	}

	rnrg := regexp.MustCompile(`(\r\n?|\n){2,}`)

	messageCreate := func(s *discordgo.Session, m *discordgo.MessageCreate) {
		if m.Author.ID == s.State.User.ID {
			return
		}

		ctx := context.Background()
		prompt := fmt.Sprintf(setup, rules, m.Content)
		res, err := service.Evaluate(ctx, "gpt3p5", prompt)
		if err != nil {
			fmt.Println("error evaluating message,", err)
			return
		}

		if strings.Trim(res, ". ") == "OK" {
			color.Green("message '%s' is OK", m.Content)
			// fmt.Printf("message '%s' is OK\n", m.Content)
			return
		} else {
			color.Red("message '%s' is _NOT_ OK", m.Content)
			// fmt.Printf("message '%s' is _NOT_ OK\n", m.Content)
		}

		// remove subsequent lines
		res = rnrg.ReplaceAllString(res, "\n")

		_, err = s.ChannelMessageSendReply(m.ChannelID, res, m.Reference())
		if err != nil {
			fmt.Println("error sending message,", err)
			return
		}
	}

	// Register the messageCreate func as a callback for MessageCreate events.
	dg.AddHandler(messageCreate)

	// In this example, we only care about receiving message events.
	dg.Identify.Intents = discordgo.IntentsGuildMessages

	// Open a websocket connection to Discord and begin listening.
	err = dg.Open()
	if err != nil {
		fmt.Println("error opening connection,", err)
		return
	}

	// Wait here until CTRL-C or other term signal is received.
	fmt.Println("Bot is now running.  Press CTRL-C to exit.")
	sc := make(chan os.Signal, 1)
	signal.Notify(sc, syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	<-sc

	// Cleanly close down the Discord session.
	dg.Close()
}
