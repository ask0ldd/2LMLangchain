import {LlamaModel, LlamaEmbeddingContext} from "node-llama-cpp"
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib"

const model = new LlamaModel({modelPath:"g:/AI/phi-2.Q5_K_M.gguf", threads:3, gpuLayers:16})

const embeddingContext = new LlamaEmbeddingContext({
    model,
    contextSize: Math.min(4096, model.trainContextSize)
})

const text = "Hello world"
const embedding = await embeddingContext.getEmbeddingFor(text)

const vectorStore = new HNSWLib()
vectorStore.addVectors(embedding)

console.log(text, embedding.vector)