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
- You MUST copy the chosen object **verbatim** from the provided context.
- DO NOT create, modify, paraphrase, rename, or invent any category.
- DO NOT remove or alter any field values.
- Matching hints can be found in each category's "regex" field. Use case-insensitive matching.
- If multiple regexes match, choose the one with the **longest regex pattern** (most specific).
- If no regex matches or none clearly fits, return the category object whose name is **"Uncategorized"**.
- If uncertain for any reason, always choose "Uncategorized".

### OUTPUT FORMAT:
- Return ONLY a single JSON object (not an array).
- Include only the keys: name, description, icon, type.
- No markdown, no explanations, no extra text.

### EXAMPLE:
Payment description: "000000095152 HEAD OFFICE BRANCH -IBTC PLACE WEB PURCHASE @SPOTIFY_VHVLOU175640333"

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
  input: "000000049337 HEAD OFFICE BRANCH -IBTC PLACE WEB PURCHASE @UDEMY LAGOS NG NG",
});
console.log("✅ Response:", response);



