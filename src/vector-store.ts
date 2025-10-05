import { TypeORMVectorStore } from "@langchain/community/vectorstores/typeorm";
import { OpenAIEmbeddings } from "@langchain/openai";
import { DataSource, DataSourceOptions } from "typeorm";
import crypto from "crypto";
import { Document } from "@langchain/core/documents";

export class VectorStore {
  private readonly dbOptions: DataSourceOptions;

  constructor() {
    this.dbOptions = {
      type: process.env.DB_TYPE as any,
      host: process.env.DB_HOST,
      port: parseInt(process.env.DB_PORT as string),
      username: process.env.DB_USERNAME,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
    } as DataSourceOptions;
  }

  /** Compute hash of all category docs to detect changes */
  private computeHash(data: any): string {
    return crypto.createHash("sha256").update(JSON.stringify(data)).digest("hex");
  }

  /** Initialize vector store and retriever */
  async getRetriever(docs: Document[]) {
    // Step 1: Initialize TypeORM data source
    const dataSource = new DataSource(this.dbOptions);
    await dataSource.initialize();

    // Step 2: Initialize vector store with explicit config object (typed)
    const vectorStore = await TypeORMVectorStore.fromDataSource(
      new OpenAIEmbeddings(),
      {
        postgresConnectionOptions: this.dbOptions, // ✅ satisfies type definition
        tableName: "langchain_pg_embedding", // optional, explicit naming
      }
    );

    // Step 3: Ensure embedding table exists
    await vectorStore.ensureTableInDatabase();

    // Step 4: Create metadata table if missing (for hash tracking)
    await dataSource.query(`
      CREATE TABLE IF NOT EXISTS vector_metadata (
        key TEXT PRIMARY KEY,
        value TEXT
      );
    `);

    // Step 5: Compute hash of current docs
    const currentHash = this.computeHash(docs.map((d) => d.pageContent));

    // Step 6: Fetch stored hash (if any)
    const existingHashResult = await dataSource.query(
      `SELECT value FROM vector_metadata WHERE key = 'categories_hash' LIMIT 1;`
    );
    const storedHash = existingHashResult?.[0]?.value ?? null;

    // Step 7: Check if embedding table exists and has data
    const tableExistsResult = await dataSource.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_name = 'langchain_pg_embedding'
      );
    `);
    const tableExists = tableExistsResult?.[0]?.exists || false;

    let existingCount = 0;
    if (tableExists) {
      const countResult = await dataSource.query(
        `SELECT COUNT(*) FROM langchain_pg_embedding;`
      );
      existingCount = parseInt(countResult?.[0]?.count || "0", 10);
    }

    // Step 8: Determine if embeddings need refresh
    const needsEmbedding =
      !tableExists || !existingCount || !storedHash || storedHash !== currentHash;

    if (needsEmbedding) {
      console.log("🧠 Updating vector embeddings...");

      if (tableExists) {
        await dataSource.query(`DROP TABLE IF EXISTS langchain_pg_embedding;`);
        console.log("🧹 Old embeddings cleared");
      }

      await vectorStore.ensureTableInDatabase();
      await vectorStore.addDocuments(docs);
      console.log("✅ New embeddings added");

      // Save or update new hash
      await dataSource.query(
        `
        INSERT INTO vector_metadata (key, value)
        VALUES ('categories_hash', $1)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
      `,
        [currentHash]
      );
    } else {
      console.log("✅ Embeddings unchanged — skipping re-embedding");
    }

    await dataSource.destroy();

    // Step 9: Return retriever
    return vectorStore.asRetriever();
  }
}