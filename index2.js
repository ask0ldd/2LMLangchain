import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { HumanMessage } from "@langchain/core/messages";
import bodyParser from "body-parser"
import express from "express"

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

app.post('/chat', async (req, res) => {
    console.log(new Date().getMinutes())
    const postData = req.body
    console.log(req.body)
    res.send('String received');
    let concatenatedTokens = ""
    const stream = await model.stream([new HumanMessage({ content: postData.question }),])
    for await (const chunk of stream) {
      concatenatedTokens += chunk.content
      console.log(concatenatedTokens)
    }
    console.log(new Date().getMinutes())
    console.log(response)
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