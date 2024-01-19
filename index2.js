import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { LlamaCppEmbeddings } from "@langchain/community/embeddings/llama_cpp";
// import {HtmlToTextTransformer} from "@langchain/community/document_transformers/html_to_text"
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import bodyParser from "body-parser"
import express from "express"
import cors from "cors"
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
const model = new ChatLlamaCpp({ modelPath: mixtral11b2, temperature:0.1, threads:3, contextSize:2048, batchSize:512, gpuLayers: 14 /* 18 */, maxTokens : 100, f16Kv:true/*, embedding:true*/})

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
    const stream = await model.stream([new SystemMessage({content : "You are a helpful assistant."}), new HumanMessage({ content: postData.question })])
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