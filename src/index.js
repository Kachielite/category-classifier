import { LoadDocument } from "./load-document.js";
import { DocumentSplit } from "./document-split.js";
import { VectorStore } from "./vector-store.js";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from 'dotenv';
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
dotenv.config();
// 1. Load json file
const loader = new LoadDocument('./data.json');
const docs = await loader.getLoader();
console.info(`Docs loaded: ${docs.length}`);
// 2. Split docs
const splitter = new DocumentSplit(docs);
const splitDocs = await splitter.split();
console.info(`Docs split: ${splitDocs.length}`);
// 3. Create vector store and retriever
const vectorStore = new VectorStore();
const retriever = vectorStore.getRetriever(splitDocs);
console.log("Retriever created");
// 4. Instantiate LLM
const llm = new ChatOpenAI({ temperature: 0, model: 'gpt-3.5-turbo' });
// 5. Define the schema
const categorySchema = z.object({
    name: z.string().describe("The name of the category"),
    description: z.string().describe("The description of the category"),
    icon: z.string().describe("The icon of the category"),
    type: z.enum(["income", "expense"]).describe("The type of the category either income or expense"),
});
// 6. Create prompt with structured output
const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You will receive a payment description and required to return the most suitable category as a JSON object with name, description, icon, and type fields."],
    ["human", "{input}"],
]);
// 7. Create chain with structured output
const chain = llm.pipe(prompt).pipe(retriever).pipe(categorySchema.parse);
const response = await chain.invoke({
    input: "4TH QUARTER 2025 CARD MAINT FEE-MASTERCARD"
});
console.log(response);
