import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";


export class DocumentSplit {
    private readonly doc: Document[]
    private splitter: RecursiveCharacterTextSplitter

    constructor(doc: Document[]) {
        this.doc = doc;
        this.splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200
        });
    }

    split(): Promise<Document[]> {
        return  this.splitter.splitDocuments(this.doc);
    }
}