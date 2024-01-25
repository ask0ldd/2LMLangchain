import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { formatDocumentsAsString } from "langchain/util/document";
import { RunnablePassthrough, RunnableSequence, } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import * as fs from "fs";
import {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
  } from "@langchain/core/prompts";

const mistral7bInstruct = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf";

const text = fs.readFileSync("g:/AI/state_of_the_union.txt", "utf8")
const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 })
const docs = await textSplitter.createDocuments([text])

const embeddings = new HuggingFaceInferenceEmbeddings({model : "BAAI/bge-base-en-v1.5"}) // "all-MiniLM-L6-v2"

const vectorStore = await HNSWLib.fromDocuments(docs, embeddings)
const vectorStoreRetriever = vectorStore.asRetriever()

const model = new ChatLlamaCpp({ 
    modelPath: mistral7bInstruct, 
    temperature:0.7, 
    threads:3, 
    contextSize:2048, 
    batchSize:2048, 
    gpuLayers: 16, 
    maxTokens : 100, 
    f16Kv:true/*, embedding:true*/
})

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

/*const answer = await chain.invoke( "which american forces have been mobilized to protect the NATO countries?" )
  
console.log({ answer });*/

const answer2 = await chain.invoke( "how many barrels of petroleum will be released by america?" )
  
console.log({ answer2 });



