import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
export class DocumentSplit {
    constructor(doc) {
        this.doc = doc;
        this.splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200
        });
    }
    split() {
        return this.splitter.splitDocuments(this.doc);
    }
}
