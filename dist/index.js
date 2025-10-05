import { LoadDocument } from "./load-document.js";
import { DocumentSplit } from "./document-split.js";
import { VectorStore } from "./vector-store.js";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { z } from "zod";
import * as dotenv from "dotenv";
import { StructuredOutputParser } from "langchain/output_parsers";
dotenv.config();
// 1) Load JSON file
const loader = new LoadDocument("./data.json");
const docs = await loader.getLoader();
console.info(`✅ Docs loaded: ${docs.length}`);
// 2. Split documents (each category chunk)
const splitter = new DocumentSplit(docs);
const splitDocs = await splitter.split();
console.info(`✅ Docs split: ${splitDocs.length}`);
// 3. Create Vector Store & Retriever
const vectorStore = new VectorStore();
const retriever = await vectorStore.getRetriever(splitDocs);
console.log("✅ Retriever created");
// 4. Define strict JSON schema
const categorySchema = z.object({
    name: z.string(),
    description: z.string(),
    icon: z.string(),
    type: z.enum(["income", "expense"]),
});
// 5. Create a strict system prompt
const prompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        `You are a financial transaction classifier.

You will receive:
1. A payment description.
2. A list of category documents with details (name, description, icon, type, regex).

Your task:
- Return ONLY the single best-matching category as JSON.
- If none matches, return the "Uncategorized" category.
- Your output MUST be a valid JSON object with the following structure:

{
  "name": string,
  "description": string,
  "icon": string,
  "type": "income" | "expense"
}

Do NOT include explanations or text outside the JSON. 
If uncertain, return the "Uncategorized" category.`,
    ],
    ["human", "Payment description: {paymentDescription}\n\nCategories:\n{context}"],
]);
const llm = new ChatOpenAI({
    temperature: 0,
    model: "gpt-4o-mini",
});
const outputParser = StructuredOutputParser.fromZodSchema(categorySchema);
// 6. Create the "stuff documents" chain (for merging retrieved docs into the prompt)
const documentChain = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser,
});
// 8️⃣ Create retrieval chain
const retrievalChain = await createRetrievalChain({
    retriever,
    combineDocsChain: documentChain,
});
// 9️⃣ Run a test classification
const response = await retrievalChain.invoke({
    input: "4TH QUARTER 2025 CARD MAINT FEE-MASTERCARD",
});
console.log("🏷️ Category:", response.answer);
