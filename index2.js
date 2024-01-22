import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { LlamaCppEmbeddings } from "@langchain/community/embeddings/llama_cpp";
// import {HtmlToTextTransformer} from "@langchain/community/document_transformers/html_to_text"
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import bodyParser from "body-parser"
import express from "express"
import cors from "cors"
import { UFCDatas } from "./ufcDatas.js";
/*import { LlamaContext } from "node-llama-cpp";
import { LlamaChatSession } from "node-llama-cpp";
import { LlamaChatPromptWrapper } from "node-llama-cpp";*/
// model_kwargs={"device":"cuda"}

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors())
const port = 3000;

const mistral7bInstruct = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf";
const mixtral11b2 = "g:/AI/mixtral_11bx2_moe_19b.Q4_K_M.gguf";
const model = new ChatLlamaCpp({ modelPath: mistral7bInstruct, temperature:0.1, threads:3, contextSize:2048*1.5, batchSize:2048*1.5/*512*/, gpuLayers: 12 /* 18 */, maxTokens : 100, f16Kv:true/*, embedding:true*/})

/*const model = new LlamaModel({
  modelPath: path.join("g:/", "AI", "codellama-13b.Q3_K_M.gguf")
})
const context = new LlamaContext({model, threads: 3, contextSize:2048, batchSize:512, embedding: true});
const session = new LlamaChatSession({
    context,
    promptWrapper: new LlamaChatPromptWrapper()
})*/

let isStreaming = false

/* 
middleware
app.use((req, res, next) => {
  if (isStreaming) {
    // If streaming is in progress, ignore the request
    res.status(400).send('Streaming in progress. Please wait.');
  } else {
    // If streaming is not in progress, proceed with the request
    next();
  }
})
*/

const pirateContext = new SystemMessage(
  "You are a pirate, responses must be very verbose and in pirate dialect, add 'Arr, m'hearty!' to each sentence."
)

const helpfulAssistantContext = new SystemMessage("You are a helpful assistant.")

const knowledgeContext = new SystemMessage(
  "You are a helpful assistant responding in the most concise way possible. Your response should be followed by three questions that the user could ask to get more knowledge about the related topic. Who should know about those facts : " + UFCDatas
)

const knowledgeContext2 = new SystemMessage(
  "You are a helpful assistant, reponses contains three related questions at the end, and you should know about those facts : " + UFCDatas
)

app.post('/chat', async (req, res) => {
  if(isStreaming) return res.status(400).send('Streaming in progress. Please wait.')
  isStreaming = true
  console.log(new Date())
    const postData = req.body
    console.log(req.body)
    // res.send('String received');
    res.writeHead(200, {
      'Content-Type': 'text/plain',
      'Transfer-Encoding': 'chunked'
    });
    let concatenatedTokens = ""
    const stream = await model.stream([knowledgeContext2, new HumanMessage({ content: postData.question })])
    for await (const chunk of stream) {
      concatenatedTokens += chunk.content
      console.log(concatenatedTokens)
      res.write(chunk.content);
    }
    res.end()
    isStreaming = false
    console.log(new Date())
    // console.log(concatenatedTokens)
})
  
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`)
})

/*
  AIMessage {
    lc_serializable: true,
    lc_kwargs: {
      content: 'Hello John.',
      additional_kwargs: {}
    },
    lc_namespace: [ 'langchain', 'schema' ],
    content: 'Hello John.',
    name: undefined,
    additional_kwargs: {}
  }
*/