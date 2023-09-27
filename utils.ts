// allows us to call OpenAI to embed text into vectors that we can then store and use in pinecone
import { OpenAIEmbeddingsParams } from "langchain/embeddings/openai";

//allows us to chunk large amounts of text into smaller pieces
import { RecursiveCharacterTextSplitterParams } from "langchain/text_splitter";

import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
import { timeout } from "./config";

type PineconeIndex = {
  client: any;
  indexName: string;
  vectorDimension: number;
};

// create Pinecone index
export const createPineconeIndex = async ({
  client,
  indexName,
  vectorDimension,
}: PineconeIndex) => {
  // 1. Initiate index existence check
  console.log(`Checking if index ${indexName} exists...`);
  // 2. Get list of existing indexes
  const existingIndexes = await client.listIndexes();
  // 3. Check if index does not exist, create it
  if (!existingIndexes.includes(indexName)) {
    // 4. Log index creation initiation
    console.log(`Index ${indexName} does not exist. Creating...`);
    // 5. Create index
    await client.createIndex({
      createRequest: {
        name: indexName,
        dimension: vectorDimension,
        metric: "cosine",
      },
    });
    // 6. Log index creation completion
    console.log(`Index ${indexName} created.`);
    // 7. Wait for index to be initialized (time is specified in config.ts)
    await new Promise((resolve) => setTimeout(resolve, timeout));
  } else {
    // 8. Log index already exists
    console.log(`Index ${indexName} exists.`);
  }
};
