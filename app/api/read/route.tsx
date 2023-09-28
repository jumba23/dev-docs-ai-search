import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { queryPineconeVectorStoreAndQueryLLM } from "../../../utils";
import { indexName } from "../../../config";

export async function POST(req: NextRequest) {
  const body = await req.json();

  const client = new Pinecone({
    apiKey: process.env.NEXT_PUBLIC_PINECONE_API_KEY || "",
    environment: process.env.NEXT_PUBLIC_PINECONE_ENVIRONMENT || "",
  });

  const reqObject = {
    client,
    indexName,
    body,
  };

  console.log("reqObject: ", reqObject);

  const text = await queryPineconeVectorStoreAndQueryLLM(reqObject);

  return NextResponse.json({
    data: text,
  });
}
