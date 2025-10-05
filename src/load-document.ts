import fs from "fs";
import path from "path";
import { Document } from "@langchain/core/documents";

export class LoadDocument {
  private readonly filePath: string;

  constructor(file: string) {
    this.filePath = path.resolve(file);
  }

  public async getLoader(): Promise<Document[]> {
    const raw = fs.readFileSync(this.filePath, "utf-8");
    const data = JSON.parse(raw);

    if (!Array.isArray(data)) {
      throw new Error("❌ JSON file must be an array of category objects");
    }

    const docs = data.map((category) => {
      // full JSON as a string for embedding
      const content = JSON.stringify(category, null, 2);

      return new Document({
        pageContent: content,
        metadata: {
          name: category.name,
          type: category.type,
          regex: category.regex,
        },
      });
    });

    console.info(`✅ Transformed ${docs.length} category documents for embedding`);
    return docs;
  }
}