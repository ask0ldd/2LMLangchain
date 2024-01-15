import {fileURLToPath} from "url";
import path from "path";
import {LlamaModel, LlamaContext, LlamaChatSession, LlamaChatPromptWrapper} from "node-llama-cpp";

const __dirname = path.dirname(fileURLToPath("file:///G:/"))

const model = new LlamaModel({
    modelPath: path.join(__dirname, "AI", "mistral-7b-instruct-v0.1.Q5_K_M.gguf")
});
const context = new LlamaContext({model});
const session = new LlamaChatSession({
    context,
    promptWrapper: new LlamaChatPromptWrapper() // by default, GeneralChatPromptWrapper is used
});


const q1 = "Hi there, how are you?";
console.log("User: " + q1);

const a1 = await session.prompt(q1);
console.log("AI: " + a1);


const q2 = "Summerize what you said";
console.log("User: " + q2);

const a2 = await session.prompt(q2);
console.log("AI: " + a2);