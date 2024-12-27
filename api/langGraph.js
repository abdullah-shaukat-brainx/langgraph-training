const { z } = require("zod");
const { ChatOpenAI } = require("@langchain/openai");
const {
    MessagesAnnotation,
    StateGraph,
    START,
    Command,
    interrupt,
    MemorySaver
} = require("@langchain/langgraph");
// const fs = require('fs');

function setupLangGraph() {
    const model = new ChatOpenAI({ 
        modelName: "gpt-4o",
        temperature: 0 
    });

    // Helper function to call LLM with structured output
    function callLlm(messages, targetAgentNodes) {
        const outputSchema = z.object({
            response: z.string().describe("A human readable response to the original question. Does not need to be a final response. Will be streamed back to the user."),
            goto: z.enum(["finish", ...targetAgentNodes]).describe("The next agent to call, or 'finish' if the user's query has been resolved.")
        });
        return model.withStructuredOutput(outputSchema, { name: "Response" }).invoke(messages);
    }

    async function travelAdvisor(state) {
        const systemPrompt = 
            "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). " +
            "If you need specific sightseeing recommendations, ask 'sightseeingAdvisor' for help. " +
            "If you need hotel recommendations, ask 'hotelAdvisor' for help. " +
            "If you have enough information to respond to the user, return 'finish'. " +
            "Never mention other agents by name.";

        const messages = [{ role: "system", content: systemPrompt }, ...state.messages];
        const targetAgentNodes = ["sightseeingAdvisor", "hotelAdvisor"];
        const response = await callLlm(messages, targetAgentNodes);
        const aiMsg = { role: "ai", content: response.response, name: "travelAdvisor" };

        let goto = response.goto;
        if (goto === "finish") {
            goto = "human";
        }

        return new Command({ goto, update: { messages: [aiMsg] } });
    }

    async function sightseeingAdvisor(state) {
        const systemPrompt = 
            "You are a travel expert that can provide specific sightseeing recommendations for a given destination. " +
            "If you need general travel help, go to 'travelAdvisor' for help. " +
            "If you need hotel recommendations, go to 'hotelAdvisor' for help. " +
            "If you have enough information to respond to the user, return 'finish'. " +
            "Never mention other agents by name.";

        const messages = [{ role: "system", content: systemPrompt }, ...state.messages];
        const targetAgentNodes = ["travelAdvisor", "hotelAdvisor"];
        const response = await callLlm(messages, targetAgentNodes);
        const aiMsg = { role: "ai", content: response.response, name: "sightseeingAdvisor" };

        let goto = response.goto;
        if (goto === "finish") {
            goto = "human";
        }

        return new Command({ goto, update: { messages: [aiMsg] } });
    }

    async function hotelAdvisor(state) {
        const systemPrompt = 
            "You are a travel expert that can provide hotel recommendations for a given destination. " +
            "If you need general travel help, ask 'travelAdvisor' for help. " +
            "If you need specific sightseeing recommendations, ask 'sightseeingAdvisor' for help. " +
            "If you have enough information to respond to the user, return 'finish'. " +
            "Never mention other agents by name.";

        const messages = [{ role: "system", content: systemPrompt }, ...state.messages];
        const targetAgentNodes = ["travelAdvisor", "sightseeingAdvisor"];
        const response = await callLlm(messages, targetAgentNodes);
        const aiMsg = { role: "ai", content: response.response, name: "hotelAdvisor" };

        let goto = response.goto;
        if (goto === "finish") {
            goto = "human";
        }

        return new Command({ goto, update: { messages: [aiMsg] } });
    }

    function humanNode(state) {
        const userInput = interrupt("Ready for user input.");

        let activeAgent;
        for (let i = state.messages.length - 1; i >= 0; i--) {
            if (state.messages[i].name) {
                activeAgent = state.messages[i].name;
                break;
            }
        }

        if (!activeAgent) {
            activeAgent = "travelAdvisor"; // Default to travelAdvisor if no previous agent
        }

        return new Command({
            goto: activeAgent,
            update: {
                messages: [
                    {
                        role: "human",
                        content: userInput,
                    }
                ]
            }
        });
    }

    const builder = new StateGraph(MessagesAnnotation)
        .addNode("travelAdvisor", travelAdvisor, { 
            ends: ["human", "sightseeingAdvisor", "hotelAdvisor"] 
        })
        .addNode("sightseeingAdvisor", sightseeingAdvisor, { 
            ends: ["human", "travelAdvisor", "hotelAdvisor"] 
        })
        .addNode("hotelAdvisor", hotelAdvisor, { 
            ends: ["human", "travelAdvisor", "sightseeingAdvisor"] 
        })
        .addNode("human", humanNode, { 
            ends: ["travelAdvisor", "sightseeingAdvisor", "hotelAdvisor"] 
        })
        .addEdge(START, "travelAdvisor");

        // GRAPH GENERATING CODE
    // const graph = await builder.compile().getGraphAsync();
    // const image = await graph.drawMermaidPng();
    // const arrayBuffer = await image.arrayBuffer();
    // const buffer = Buffer.from(arrayBuffer);
    // fs.writeFileSync('graph.png', buffer);
    // console.log('Graph saved as graph.png');

    const checkpointer = new MemorySaver();
    return builder.compile({ checkpointer });
}

module.exports = { setupLangGraph }; 