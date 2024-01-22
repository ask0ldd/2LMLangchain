import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { LlamaCppEmbeddings } from "@langchain/community/embeddings/llama_cpp";
// import {HtmlToTextTransformer} from "@langchain/community/document_transformers/html_to_text"
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import bodyParser from "body-parser"
import express from "express"
import cors from "cors"
import { LlamaContext, LlamaChatSession, LlamaChatPromptWrapper, LlamaModel } from "node-llama-cpp";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import { formatDocumentsAsString } from "langchain/util/document";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "@langchain/core/prompts";
// model_kwargs={"device":"cuda"}

/*import { TextLoader } from "langchain/document_loaders/fs/text";

const loader = new TextLoader("g:/AI/bestband.txt")
const doc = await loader.load()

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 10,
    chunkOverlap: 1,
})
  
const splitDatas = await splitter.splitDocuments([doc])*/

const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors())
const port = 3000;

const mistral7bInstruct = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf";
const mixtral11b2 = "g:/AI/mixtral_11bx2_moe_19b.Q4_K_M.gguf";
// const model = new ChatLlamaCpp({ modelPath: mixtral11b2, temperature:0.1, threads:3, contextSize:2048, batchSize:512, gpuLayers: 14 /* 18 */, maxTokens : 100, f16Kv:true/*, embedding:true*/})

const model = new LlamaModel({
  modelPath: mistral7bInstruct
})
const context = new LlamaContext({model, threads: 3, contextSize:2048, batchSize:512, embedding: true, });
const session = new LlamaChatSession({
    context,
    promptWrapper: new LlamaChatPromptWrapper()
})

const text = fs.readFileSync("g:/AI/bestband.txt", "utf8")
const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 })
const docs = await textSplitter.createDocuments([text])
const embeddings = new LlamaCppEmbeddings({ modelPath: mistral7bInstruct, embedding: true, })
console.log(embeddings)
const vectorStore = await HNSWLib.fromDocuments(docs, embeddings)
const vectorStoreRetriever = vectorStore.asRetriever()

const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}`;
const messages = [
  SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
  HumanMessagePromptTemplate.fromTemplate("{question}"),
];
const prompt = ChatPromptTemplate.fromMessages(messages);

const chain = RunnableSequence.from([
  {
    context: vectorStoreRetriever.pipe(formatDocumentsAsString),
    question: new RunnablePassthrough(),
  },
  prompt,
  model,
  new StringOutputParser(),
])

const answer = await chain.invoke(
  "What did the president say about Justice Breyer?"
)

console.log({ answer });

/*let isStreaming = false

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
})*/

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