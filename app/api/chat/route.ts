import { google } from '@ai-sdk/google';
import { streamText, tool } from 'ai';
import { z } from 'zod';
import { hybridRAG } from '@/lib/ai/embeddings';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
    const { messages } = await req.json();

    const result = streamText({
        model: google('gemini-2.0-flash-exp', {
            useSearchGrounding: false,
        }),
        system: `You are a helpful assistant. Check your knowledge base before answering any questions.
            Only respond to questions using information from tool calls.
            if no relevant information is found in the tool calls, respond, "Sorry, I don't know."
            Limit your responses to maximum 100 words.
            `,
        messages,
        tools: {
            getInformation: tool({
                description: `Get hybrid information (both traditional and graph-based) from your knowledge base.`,
                parameters: z.object({
                    question: z.string().describe('the user\'s question'),
                }),
                execute: async ({ question }) => {
                    const context = await hybridRAG(question);
                    return JSON.stringify(context);
                },
            }),
            // getMostRepeatedQuestion: tool({
            //     description: `Get the most repeated question from previous year question papers.`,
            //     parameters: z.object({}),
            //     execute: async () => findFreqofQuestions(),
            // }),
        },
    });

    return result.toDataStreamResponse();
}