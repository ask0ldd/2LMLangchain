import { LlamaCppEmbeddings } from "@langchain/community/embeddings/llama_cpp";
import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";

const mistral7bInstruct = "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf";

// const model = new ChatLlamaCpp({ modelPath: mistral7bInstruct, temperature:0.1, threads:3, contextSize:2048*1.5, batchSize:2048*1.5/*512*/, gpuLayers: 12 /* 18 */, maxTokens : 100, f16Kv:true, embedding:true})

const embeddings = new LlamaCppEmbeddings({
    modelPath: "g:/AI/mistral-7b-instruct-v0.1.Q5_K_M.gguf", threads : 3, f16Kv:true, embedding:true
  });
  
  // Embed a query string using the Llama embeddings
  const res = await embeddings.embedDocuments(["cat", "dog", "car", "bike"]);
  
  // .embedQuery("Hello Llama!");
  
  // Output the resulting embeddings
  console.log(res);
  