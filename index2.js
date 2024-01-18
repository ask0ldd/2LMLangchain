import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { HumanMessage } from "@langchain/core/messages";
import bodyParser from "body-parser"
import express from "express"
import cors from "cors"

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors())
const port = 3000;

const llamaPath = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf";
// const llamaPath = "g:/AI/zyte-1B-q8_0.gguf";
// const llamaPath = "g:/AI/phi-2.Q5_K_M.gguf";
const model = new ChatLlamaCpp({ modelPath: llamaPath, threads:3, contextSize:1024, /*batchSize:512,*/ gpuLayers: 18 /* 20 */, maxTokens : 100, f16Kv:true, /*n_gpu_layers: 12, n_batch: 512, streaming: true, runManager: {
  handleLLMNewToken(token){
    process.stdout.write(token)
    console.log(token)
  },
}*/})

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
    const stream = await model.stream([new HumanMessage({ content: postData.question }),])
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