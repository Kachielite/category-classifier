import * as dotenv from 'dotenv';
import { TypeORMVectorStore } from "@langchain/community/vectorstores/typeorm";
import { OpenAIEmbeddings } from "@langchain/openai";
dotenv.config();
export class VectorStore {
    constructor() {
        this.args = {
            postgresConnectionOptions: {
                type: process.env.DB_TYPE,
                host: process.env.DB_HOST,
                port: parseInt(process.env.DB_PORT),
                username: process.env.DB_USERNAME,
                password: process.env.DB_PASSWORD,
                database: process.env.DB_NAME,
            },
        };
    }
    async getRetriever(doc) {
        //
        const vectorStore = await TypeORMVectorStore.fromDataSource(new OpenAIEmbeddings(), this.args);
        await vectorStore.ensureTableInDatabase();
        await vectorStore.addDocuments(doc);
        return vectorStore.asRetriever();
    }
}
