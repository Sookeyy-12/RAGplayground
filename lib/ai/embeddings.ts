import { embedMany, embed } from 'ai';
import { google } from "@ai-sdk/google";
import { DataAPIClient } from "@datastax/astra-db-ts";
import dotenv from 'dotenv';
import neo4j from 'neo4j-driver';

dotenv.config();

const ASTRA_DB_NAMESPACE = "default_keyspace";
const ASTRA_DB_COLLECTION = "RAGall";
const ASTRA_DB_API_ENDPOINT = "https://0e8d6974-1f34-4530-983e-f4bc69e0307e-us-east-2.apps.astra.datastax.com";
const ASTRA_DB_APPLICATION_TOKEN = "AstraCS:mwNtQeDfsAQobIZdFMDokjZZ:76832d0869174c963dda7b6961fecf6ff9601eaba276092f58204538c3772d8e";

const embeddingModel = google.textEmbeddingModel('text-embedding-004');

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE });

const neo4jURI = process.env.NEO4J_URI || "bolt://localhost:7687";
const neo4jUser = process.env.NEO4J_USER || "neo4j";
const neo4jPassword = process.env.NEO4J_PASSWORD || "password";
const driver = neo4j.driver(neo4jURI, neo4j.auth.basic(neo4jUser, neo4jPassword));

const generateChunks = (input: string) => {
    return input
    .trim()
    .split('.')
    .filter(i => i !== '')
};

export const generateEmbeddings = async (
    value: string,
): Promise<Array<{ embedding: number[]; content: string}>> => {
    const chunks = generateChunks(value);
    const { embeddings } = await embedMany({
        model: embeddingModel,
        values: chunks,
    });
    return embeddings.map((e, i) => ({ content: chunks[i], embedding: e}));
};

export const generateEmbedding = async (value: string): Promise<number[]> => {
    const input = value.replaceAll('\\n', ' ');
    const { embedding } = await embed({
        model: embeddingModel,
        value: input,
    });
    return embedding;
};

export const findRelevantContent = async ( userQuery: string) => {
    const userQueryEmbed = await generateEmbedding(userQuery);
    let docContext = "";
    try {
        const collection = await db.collection(ASTRA_DB_COLLECTION);
        const cursor = collection.find({}, {
            sort: {
                $vector: userQueryEmbed,
            },
            limit: 10
        });
        const documents = await cursor.toArray();
        const docsMap = documents?.map(doc => doc.text);
        docContext = JSON.stringify(docsMap);
    } catch (error) {
        console.log("Error finding relevant content", error);
        docContext = "";
    };
    return docContext;
};

export const findFreqofQuestions = async () => {
    let docContext = "";
    try {
        const collection = await db.collection(ASTRA_DB_COLLECTION);
        const cursor = collection.find({});
        const documents = await cursor.toArray();
        const docsMap = documents?.map(doc => doc.text);

        // Find the most repeated questions
        const questionFrequency: { [key: string]: number } = {};
        docsMap.forEach(question => {
            questionFrequency[question] = (questionFrequency[question] || 0) + 1;
        });

        // Sort questions by frequency and get the top 3-5 questions
        const sortedQuestions = Object.keys(questionFrequency).sort((a, b) => questionFrequency[b] - questionFrequency[a]);
        const mostRepeatedQuestions = sortedQuestions.slice(0, 5);

        docContext = JSON.stringify({ mostRepeatedQuestions, allQuestions: docsMap });
    } catch (error) {
        console.log("Error finding relevant content", error);
        docContext = "";
    }
    return docContext;
};

export const findGraphRelevantContent = async (userQuery: string): Promise<string> => {
    const userQueryEmbed = await generateEmbedding(userQuery);
    const session = driver.session();
    let docsMap: string[] = [];
    try {
        // Retrieve nodes from Neo4j
        const result = await session.run(
            "MATCH (n:Document) RETURN n.text as text, n.embedding as embedding"
        );
        // Calculate cosine similarity
        const cosineSimilarity = (a: number[], b: number[]) => {
            const dot = a.reduce((sum, cur, idx) => sum + (cur * (b[idx] || 0)), 0);
            const normA = Math.sqrt(a.reduce((sum, cur) => sum + cur * cur, 0));
            const normB = Math.sqrt(b.reduce((sum, cur) => sum + cur * cur, 0));
            return normA && normB ? dot / (normA * normB) : 0;
        };
        // Map records & sort by similarity
        const records = result.records.map(rec => ({
            text: rec.get("text"),
            embedding: rec.get("embedding")
        }));
        records.sort((a, b) => cosineSimilarity(b.embedding, userQueryEmbed) - cosineSimilarity(a.embedding, userQueryEmbed));
        docsMap = records.slice(0, 10).map(r => r.text);
    } catch (error) {
        console.log("Error finding graph relevant content", error);
    } finally {
        await session.close();
    }
    return JSON.stringify(docsMap);
};

export const hybridRAG = async (userQuery: string): Promise<{ traditional: string; graph: string }> => {
    const traditional = await findRelevantContent(userQuery);
    const graph = await findGraphRelevantContent(userQuery);
    return { traditional, graph };
};