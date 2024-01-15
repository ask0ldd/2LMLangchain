import { LLMChain } from "langchain/chains";
import bodyParser from "body-parser"
import express from "express"
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
// import { HumanMessage } from "@langchain/core/messages";

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
const port = 3000;

const llamaPath = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf";
const model = new ChatLlamaCpp({ modelPath: llamaPath, gpuLayers: 24, /*n_gpu_layers: 12,*/ n_batch: 512, streaming: true, runManager: {
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
console.log(new Date())
const resB = await chainB.call({
  input_language: "English",
  output_language: "French",
  text: `This will create a new Date object with the current date and time, and the console.log will display it in the console.`,
})
/*let concatenatedTokens = ""
const stream = await chainB.stream({
  input_language: "English",
  output_language: "French",
  text: `This will create a new Date object with the current date and time.`,
})
for await (const chunk of stream) {
  concatenatedTokens += chunk.content
  console.log(concatenatedTokens)
}*/
console.log({ resB });
console.log(new Date())
// { resB: { text: "J'adore la programmation." } }