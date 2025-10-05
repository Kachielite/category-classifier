import { JSONLoader } from "langchain/document_loaders/fs/json";
export class LoadDocument {
    constructor(file) {
        this.loader = new JSONLoader(file);
    }
    getLoader() {
        return this.loader.load();
    }
}
