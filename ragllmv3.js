import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp"
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib"
import { formatDocumentsAsString } from "langchain/util/document"
import { RunnablePassthrough, RunnableSequence, } from "@langchain/core/runnables"
import { StringOutputParser } from "@langchain/core/output_parsers"
import * as fs from "fs"
import { PDFLoader } from "langchain/document_loaders/fs/pdf"
import { ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, } from "@langchain/core/prompts"

const mistral7bInstruct = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf"

async function fileToSplitDocs(filename){
  const text = fs.readFileSync(filename, "utf8")
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1500, chunkOverlap: 200, separators : ' ' })
  const docs = await textSplitter.createDocuments([text])
  return docs
}

async function pdfToSplitDocs(filename){
  const loader = new PDFLoader(filename, {
    parsedItemSeparator: " ",
  })
  const docs = await loader.load()
  return docs
}

async function webPageToSplitDocs(url){

}

const docs = await fileToSplitDocs("g:/AI/state_of_the_union.txt")
const docs2 = await fileToSplitDocs("g:/AI/montecristo-chapter1.txt")
const docs3 = await pdfToSplitDocs("g:/AI/llamaPaper.pdf")

const embeddings = new HuggingFaceInferenceEmbeddings({model : "BAAI/bge-base-en-v1.5"}) // "all-MiniLM-L6-v2"

const vectorStore = await HNSWLib.fromDocuments(docs.concat(docs2)/*.concat(docs3)*/, embeddings)
const vectorStoreRetriever = vectorStore.asRetriever()

const model = new ChatLlamaCpp({ 
    modelPath: mistral7bInstruct, 
    temperature:0.7, 
    threads:3, 
    contextSize:2048, 
    // batchSize:2048, 
    batchSize:1024,
    gpuLayers: 16, 
    maxTokens : 2048, 
    f16Kv:true/*, embedding:true*/
})

const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}`
const messages = [
  SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
  HumanMessagePromptTemplate.fromTemplate("{question}"),
]
const prompt = ChatPromptTemplate.fromMessages(messages)

const chain = RunnableSequence.from([
    {
      context: vectorStoreRetriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
])

const questions = {
  barrels : "how many barrels of petroleum will be released by america?",
  mobilization : "which american forces have been mobilized to protect the NATO countries?",
  banks : "what is happening to russian's banks?",
  financial : "what is happening to russian's financial actors?",
  monte : "how much time will it takes to get back to sea after emptying the cargo?",
  monte2 : "who has to carry a packet and a letter for Leclere?",
  llama : "what is LLaMA?",
}

const answer = await chain.invoke(questions.monte)
  
console.log({ answer })



