import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from 'dotenv';
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
dotenv.config();
// Instantiate LLM
const llm = new ChatOpenAI({ temperature: 0, model: 'gpt-3.5-turbo' });
// Define the schema
const categorySchema = z.object({
    name: z.string().describe("The name of the category"),
    description: z.string().describe("The description of the category"),
    icon: z.string().describe("The icon of the category"),
    type: z.enum(["income", "expense"]).describe("The type of the category either income or expense"),
});
// Create prompt with structured output
const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You will receive a payment description and required to return the most suitable category as a JSON object with name, description, icon, and type fields."],
    ["human", "{paymentDescription}"],
]);
// Create chain with structured output
const structuredLlm = llm.withStructuredOutput(categorySchema);
const chain = prompt.pipe(structuredLlm);
const response = await chain.invoke({
    paymentDescription: "4TH QUARTER 2025 CARD MAINT FEE-MASTERCARD"
});
console.log(response);
