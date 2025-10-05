# Category Classifier

A financial transaction classifier that uses vector embeddings and OpenAI to strictly assign transactions to predefined categories. The classifier will only select from the provided categories in `src/data.json`, and if no match is found, it will assign the transaction to "Uncategorized".

## Features
- Strict category selection from a JSON list
- No category invention or modification
- Uses OpenAI embeddings and TypeORM vector store
- PostgreSQL database backend
- Easily extensible category definitions

## Prerequisites
- Node.js >= 18
- PostgreSQL database
- OpenAI API key

## Installation
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd category-classifier
   ```
2. Install dependencies:
   ```sh
   npm install
   ```

## Configuration
Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
DB_TYPE=postgres
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
```

## Usage
To run the classifier:
```sh
npm start
```
Or for development mode:
```sh
npm run dev
```

The main entry point is `src/index.ts`. It loads categories from `src/data.json`, initializes the vector store, and sets up the classifier.

## Category Selection Logic
- The classifier receives a payment description and the list of categories.
- It must select **exactly one** category object from the provided list.
- If no match is found, it selects:
  ```json
  {
    "name": "Uncategorized",
    "description": "Transactions that do not fit into any predefined category or require manual review.",
    "icon": "❓",
    "type": "expense",
    "regex": null
  }
  ```
- **No new categories are invented or modified.**

## Troubleshooting
- Ensure your PostgreSQL database is running and accessible.
- Check your `.env` file for correct credentials.
- If embeddings are not updating, delete the `langchain_pg_embedding` table in your database and restart.

## Extending Categories
Edit `src/data.json` to add, remove, or modify categories. The classifier will only use categories defined in this file.

