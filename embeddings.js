import {LlamaModel, LlamaEmbeddingContext, LlamaEmbedding, LlamaContext} from "node-llama-cpp"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib"
import * as fs from "fs"

async function fileToSplitDocs(filename){
    const text = fs.readFileSync(filename, "utf8")
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1500, chunkOverlap: 200, separators : ' ' })
    const docs = await textSplitter.createDocuments([text])
    return docs
  }

const model = new LlamaModel({modelPath:"g:/AI/phi-2.Q5_K_M.gguf", threads:3, gpuLayers:33})

const embeddingContext = new LlamaEmbeddingContext({
    model,
    contextSize: Math.min(4096, model.trainContextSize)
})

/*const context = new LlamaContext({
    model,
    contextSize: Math.min(4096, model.trainContextSize)
})

context.e*/

// const embeddings = new LlamaEmbedding()
const LlamaEmb = {
    _model : model,
    _context : embeddingContext,
    embedQuery : async (text) => embeddingContext.getEmbeddingFor(text),
    embedDocuments : async (texts) => {
       // await texts.map(text => embeddingContext.getEmbeddingFor(text))
            const embeddings = []
            for(let i=0; i<texts.length; i++)
            {
                embeddings.push(await embeddingContext.getEmbeddingFor(texts[i]))
            }
            return embeddings
    },
}

const text = "Hello world"
// const embedding = await embeddingContext.getEmbeddingFor(text)

console.log(text, await LlamaEmb.embedQuery(text)/*embedding.vector*/)

// const vectorStore = await HNSWLib.fromTexts([text], LlamaEmb)

const docs = await fileToSplitDocs("g:/AI/state_of_the_union3.txt")
const vectorStore = await HNSWLib.fromDocuments(docs/*.concat(docs2).concat(docs3)*/, LlamaEmb)

// vectorStore.addVectors(embedding.vector, text)

/*
export declare class LlamaCppEmbeddings extends Embeddings {
    _model: LlamaModel;
    _context: LlamaContext;
    constructor(inputs: LlamaCppEmbeddingsParams);
    
     * Generates embeddings for an array of texts.
     * @param texts - An array of strings to generate embeddings for.
     * @returns A Promise that resolves to an array of embeddings.
    
    embedDocuments(texts: string[]): Promise<number[][]>;
    
     * Generates an embedding for a single text.
     * @param text - A string to generate an embedding for.
     * @returns A Promise that resolves to an array of numbers representing the embedding.
    
    embedQuery(text: string): Promise<number[]>;
}
*/