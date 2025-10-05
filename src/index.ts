import { LoadDocument } from "./load-document.js";
// import { DocumentSplit } from "./document-split.js";
import { VectorStore } from "./vector-store.js";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import * as dotenv from "dotenv";

dotenv.config();

// 1) Load JSON file (DOCUMENT & EMBEDDING PIPELINE UNCHANGED)
const loader = new LoadDocument("./src/data.json");
const docs = await loader.getLoader();
console.info(`✅ Docs loaded: ${docs.length}`);

// 2. Split documents (UNCHANGED)
// const splitter = new DocumentSplit(docs);
// const splitDocs = await splitter.split();
// console.info(`✅ Docs split: ${splitDocs.length}`);
// Skip splitting because each category is atomic, If you ever embed large descriptions or multi-paragraph texts later, you can bring back the RecursiveCharacterTextSplitter.

// 3. Create Vector Store & Retriever (UNCHANGED)
const vectorStore = new VectorStore();
const retriever = await vectorStore.getRetriever(docs);
console.log("✅ Retriever created");


// 5. Create a strict system prompt (reinforced rules)
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a financial transaction classifier.

You will receive:
1. A payment description.
2. A list of category objects in JSON format (the "context"). Each object has:
   - name
   - description
   - icon
   - type
   - regex

Your goal is to choose **exactly one** category object from this provided list that best matches the payment description.

### CRITICAL RULES:
1. You MUST copy the chosen object **verbatim** from the provided context.
2. DO NOT create, modify, paraphrase, rename, or invent any category.
3. DO NOT remove or alter any field values.

### CLASSIFICATION LOGIC:
1. First, attempt to match the payment description using the "regex" field of each category (case-insensitive).
2. If multiple regex patterns match, choose the one with the **longest** pattern (most specific).
3. If no regex match is found:
   - Check if the payment description contains **human names** — multiple capitalized words like "JOHN DOE" or "EMILY CARTER".
   - If it looks like a personal name or includes sender/receiver details, classify it as **"Peer-to-Peer Transfer"** if that category exists.
4. If no suitable category is found, return the one named **"Uncategorized"**.

### OUTPUT FORMAT:
- Return ONLY a single JSON object (not an array).
- Include only the keys: name, description, icon, type.
- No markdown, no explanations, no extra text.

### EXAMPLE:
Payment description: "RVSL/WEB PYMT JETBRAINS PRAGUE CZ/VSH/(29-"
Context includes:
{{
  "name": "Subscriptions",
  "description": "Recurring payments to digital services like music, video streaming, SaaS tools, or online memberships.",
  "icon": "📺",
  "type": "expense",
  "regex": "(SPOTIFY|NETFLIX|APPLE|MICROSOFT|GOOGLE|YOUTUBE|DISNEY|SUBSCRIPTION|PLAN|PREMIUM)"
}}

Expected Output:
{{
  "name": "Subscriptions",
  "description": "Recurring payments to digital services like music, video streaming, SaaS tools, or online memberships.",
  "icon": "📺",
  "type": "expense"
}}`,
  ],
  [
    "human",
    `Payment description: "{input}"

Retrieved context:
{context}

Return exactly ONE object copied verbatim from the context.
If unsure, return the one whose "name" is "Uncategorized".`,
  ],
]);

// 6. Model (temperature lowered to reduce drift)
const llm = new ChatOpenAI({
  temperature: 0,
  model: "gpt-4o-mini",
});

// 6b. Stuff documents chain (UNCHANGED MECHANICALLY)
const documentChain = await createStuffDocumentsChain({
  llm,
  prompt,
});

// 7. Retrieval chain (UNCHANGED)
const retrievalChain = await createRetrievalChain({
  retriever,
  combineDocsChain: documentChain,
});

// 8. Query the chain
const response = await retrievalChain.invoke({
  input: "ALS2161135274JS_Tappi Inc.",
});
console.log("✅ Response:", response);



