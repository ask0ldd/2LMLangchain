import { LLMChain } from "langchain/chains";
import bodyParser from "body-parser"
import express from "express"
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { HumanMessage } from "@langchain/core/messages";

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
const port = 3000;

const llamaPath = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf";
const model = new ChatLlamaCpp({ modelPath: llamaPath, n_gpu_layers: 12, n_batch: 512, streaming: true, runManager: {
  handleLLMNewToken(token){
    process.stdout.write(token)
    console.log(token)
  },
}})

const chatPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a helpful assistant that translates {input_language} to {output_language}.",
  ],
  ["human", "{text}"],
]);
const chainB = new LLMChain({
  prompt: chatPrompt,
  llm: model,
});

const resB = await chainB.call({
  input_language: "English",
  output_language: "French",
  text: "I love programming.",
});
console.log({ resB });
// { resB: { text: "J'adore la programmation." } }