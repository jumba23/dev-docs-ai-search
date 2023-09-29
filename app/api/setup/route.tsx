import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { createPineconeIndex, updatePineconeIndex } from "../../../utils";
import { indexName } from "../../../config";

export async function POST() {
  const loader = new DirectoryLoader("./documents", {
    ".txt": (path) => new TextLoader(path),
    ".md": (path) => new TextLoader(path),
    ".pdf": (path) => new PDFLoader(path),
  });

  const docs = await loader.load();
  const vectorDimensions = 1536;

  const client = new Pinecone({
    apiKey: process.env.NEXT_PUBLIC_PINECONE_API_KEY || "",
    environment: process.env.NEXT_PUBLIC_PINECONE_ENVIRONMENT || "",
  });

  let responseMessage = "An unknown error has occurred";

  try {
    await createPineconeIndex({
      client: client,
      indexName: indexName,
      vectorDimension: vectorDimensions,
    });
    await updatePineconeIndex({
      client: client,
      indexName: indexName,
      docs: docs,
    });

    responseMessage =
      "successfully created index and loaded data into pinecone...";
    return NextResponse.json({ data: responseMessage });
  } catch (err) {
    console.error("Error:", err);

    responseMessage = "Failed to create index or load data into pinecone";
    return NextResponse.json({ data: responseMessage, status: 500 });
  }
}
