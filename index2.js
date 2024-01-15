import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { LLMChain } from "langchain/chains";
import {ChatPromptTemplate} from "langchain/prompts"
// import { StreamingHandler } from "@langchain/core/utils/stream"
/*import { SystemMessage } from "@langchain/core/messages";
import { CallbackHandler } from "langfuse-langchain";*/
import { HumanMessage } from "@langchain/core/messages";
import bodyParser from "body-parser"
import express from "express"

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
const port = 3000;

// Set up the streaming handler
/*const handler = new StreamingHandler((tokens) => {
  // Process the tokens in real time
  console.log(tokens);
});*/

function logger(tokens){
  console.log(tokens)
}

const llamaPath = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf";
const model = new ChatLlamaCpp({ modelPath: llamaPath, n_gpu_layers: 12, n_batch: 512, streaming: true,})

app.post('/chat', async (req, res) => {
    console.log(new Date().getMinutes())
    const postData = req.body
    console.log(req.body)
    res.send('String received');
    /*const promptTemplate = new ChatPromptTemplate('Your prompt message here');
    const chain = new LLMChain({ llm: model })
    chain.stream(promptTemplate, handler);*/
    const response = await model.invoke([new HumanMessage({ content: postData.question }),], undefined, [
      {
        handleLLMNewToken(token){
          process.stdout.write(token)
          console.log(token)
        }
      }      
    ])
    console.log(new Date().getMinutes())
    console.log(response)
})
  
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`)
})

/*console.log(new Date().getMinutes())
const response = await model.invoke([
  new HumanMessage({ content: "My name is John." }),
]);
console.log({ response });
console.log(new Date().getMinutes())

const response2 = await model.invoke([
    new HumanMessage({ content: "Tell me who is the most famous french writer." }),
  ]);
console.log({ response2 });
console.log(new Date().getMinutes())
*/
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