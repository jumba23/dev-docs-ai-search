// allows us to call OpenAI to embed text into vectors that we can then store and use in pinecone
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

//allows us to chunk large amounts of text into smaller pieces
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
import { timeout } from "./config";

type PineconeIndex = {
  client: any;
  indexName: string;
  vectorDimension?: number;
  docs?: Document[];
  question?: string;
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

// function that loads data into Pinecone index
export const updatePineconeIndex = async ({
  client,
  indexName,
  docs,
}: PineconeIndex) => {
  // 1.  Retrieve Pinecone index
  const index = client.getIndex({ indexName });
  // 2. Log the retrieval of the index
  console.log(`Retrieved index ${indexName}`);
  // 3. Process each document in the docs array
  if (docs) {
    for (const doc of docs) {
      console.log(`Processing document ${doc.metadata.source}`);
      // path on the local file system to the document
      const txtPath = doc.metadata.source;
      // text content of the document
      const text = doc.pageContent;
      // 4. Create RecursiveCharacterTextSplitterParams instance
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
      });
      console.log(`Splitting text into chunks...`);
      // 5. Split text into chunks (documents)
      const chunks = await textSplitter.createDocuments([text]);
      console.log(`Text split into ${chunks.length} chunks.`);
      console.log(
        `Calling OpenAI's Embedding endpoint documents with ${chunks.length} text chunks...`
      );
      // 6. Create OpenAI embeddings for documents
      const embeddingsArrays = await new OpenAIEmbeddings().embedDocuments(
        chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
      );
      console.log(
        `Creating ${chunks.length} vectors array with id, values, and metadata...`
      );

      // 7. Create and upsert vectors in batches of 100
      const batchSize = 100;
      let batch: any = [];
      for (let idx = 0; idx < chunks.length; idx++) {
        const chunk = chunks[idx];
        // 8. Create vector object
        const vector = {
          id: `${txtPath}_${idx}    `,
          values: embeddingsArrays[idx],
          metadata: {
            ...chunk.metadata,
            loc: JSON.stringify(chunk.metadata.loc),
            pageContent: chunk.pageContent,
            txtPath: txtPath,
          },
        };
        // 9. Add vector to batch
        batch.push(vector);
        // 10. If batch is full, upsert vectors
        if (batch.length === batchSize) {
          console.log(`Upserting ${batch.length} vectors...`);
          await index.upsertVectors({ vectors: batch });
          console.log(`Upserted ${batch.length} vectors.`);
          // 11. Push vector to batch
          batch = [...batch, vector];
          // When batch is full or it's the last item, upsert the vectors
          if (batch.length === batchSize || idx === chunks.length - 1) {
            console.log(`Upserting ${batch.length} vectors...`);
            await index.upsert({ vectors: batch });
            console.log(`Upserted ${batch.length} vectors.`);
            // 12. Reset batch
            batch = [];
          }
        }
      }
    }
  }
};

// function that queries Pinecone vector store AND queries LLM

export const queryPineconeVectorStoreAndQueryLLM = async ({
  client,
  indexName,
  question,
}: PineconeIndex) => {
  // 1. Start query process
  console.log("Starting query Pinecone vector store...");
  // 2. Retrieve the Pinecone index
  const index = client.getIndex({ indexName });
  // 3. Create query embedding
  if (question) {
    const queryEmbedding = await new OpenAIEmbeddings().embedDocuments([
      question,
    ]);
    // 4. Query Pinecone index and return top 10 matches
    let queryResponse = await index.query({
      queryRequest: {
        vector: queryEmbedding,
        topK: 10,
        includeMetadata: true,
        includeValues: true,
      },
    });
    // 5. Log the number of matches
    console.log(`Found ${queryResponse.results.length} matches.`);
    // 6. Log the questions being asked
    console.log(`Asking question: ${question} ...`);
    if (queryResponse.matches.length) {
      // 7. Create an OpenAI instance and load the QAStuffChain
      const llm = new OpenAI();
      const chain = loadQAStuffChain(llm);
      // 8. Extract and concatenate the text from the top 10 matches
      const concatenatePageContent = queryResponse.matches
        .map((match: any) => match.metadata.pageContent)
        .join(" ");
      const result = await chain.call({
        input_documents: [
          new Document({ pageContent: concatenatePageContent }),
        ],
        question: question,
      });
      // 9. Log the answer
      console.log(`Answer: ${result.text}`);
      return result.text;
    } else {
      // 10. Log that there are no matches, so GPT will not be called
      console.log(`No matches found, so GPT will not be called.`);
      return null;
    }
  }
};
