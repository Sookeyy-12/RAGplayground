import { google } from '@ai-sdk/google';
import { streamText, tool } from 'ai';
import { z } from 'zod';
import { findRelevantContent } from '@/lib/ai/embeddings';

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
            // addResource: tool({
            //     description: `add a resource to your knowledge base.
            //     If the user provides a random piece of knowledge unprompted, use this tool without asking
            //     for confirmation.`,
            //     parameters: z.object({
            //         content: z
            //             .string()
            //             .describe('The content or resource to add to knowledge base.'),
            //     }),
            //     execute: async ({ content }) => createResource({ content }),
            // }),
            getInformation: tool({
                description: `Get information from your knowledge base to user queries.`,
                parameters: z.object({
                    question: z.string().describe('the users question'),
                }),
                execute: async ({ question}) => findRelevantContent(question),
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