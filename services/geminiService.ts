import { GoogleGenAI } from "@google/genai";
import type { NewsArticle } from '../types';

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable not set");
}
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

interface GenerationOptions {
    topics: string[];
    industry: string;
    companyName: string;
    tone: string;
    newsFormat: 'paragraph' | 'bullets';
    wordLength: number;
    additionalInstructions: string;
}

export const fetchTrendingNews = async (topics: string[], minArticles: number): Promise<NewsArticle[]> => {
  const prompt = `Find the most recent (within the last 48 hours), top trending news articles for the following topics: ${topics.join(', ')}. 
  Focus on significant developments and announcements. For each article, find its title, a brief summary, the source website, a direct link, and its publication date. 
  Return at least ${minArticles} diverse articles in total across all topics. Ensure the links are valid.
  Format the output as a valid JSON array of objects, where each object has keys: "title", "summary", "source", "link", and "date".
  Do not include any introductory text, closing text, or markdown formatting like \`\`\`json. The entire response should be only the JSON array.`;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        tools: [{ googleSearch: {} }],
      },
    });

    const rawText = response.text.trim();
    
    // The model might return markdown fences or other text around the JSON array.
    // We need to extract the JSON array robustly.
    let jsonText = rawText;

    const markdownMatch = jsonText.match(/```(?:json)?\s*(\[[\s\S]*\])\s*```/);
    if (markdownMatch && markdownMatch[1]) {
        // If JSON is inside a markdown block, extract it.
        jsonText = markdownMatch[1];
    } else {
        // Otherwise, find the first and last brackets.
        const jsonStartIndex = jsonText.indexOf('[');
        const jsonEndIndex = jsonText.lastIndexOf(']');

        if (jsonStartIndex === -1 || jsonEndIndex === -1 || jsonEndIndex < jsonStartIndex) {
            console.error("Could not find a valid JSON array in the response from Gemini. Response text:", rawText);
            throw new Error("The AI returned data in an unexpected format. Could not find a JSON array.");
        }
        jsonText = jsonText.substring(jsonStartIndex, jsonEndIndex + 1);
    }

    const articles = JSON.parse(jsonText) as NewsArticle[];
    if (!articles || articles.length === 0) {
      throw new Error("The AI returned no articles. The response might be empty or in an unexpected format.");
    }
    return articles;
  } catch (error) {
    console.error("Error fetching news from Gemini:", error);
    if (error instanceof SyntaxError) {
        throw new Error("Failed to parse the news data from the AI. The format was invalid.");
    }
    throw new Error("Failed to fetch news from the AI. Check API key or try again later.");
  }
};

export const generateNewsletter = async (articles: NewsArticle[], options: GenerationOptions): Promise<string> => {
    const { industry, companyName, tone, newsFormat, wordLength, additionalInstructions } = options;
    const articlesJson = JSON.stringify(articles, null, 2);

    const prompt = `
    You are an expert content creator for "${companyName}", a leading voice in the ${industry} sector.
    Your task is to generate a professional newsletter.
    The audience is savvy and expects high-quality, relevant information.

    **Newsletter Specifications:**
    - **Company:** ${companyName}
    - **Industry:** ${industry}
    - **Tone:** ${tone}. The writing must reflect this tone consistently.
    - **News Format:** Each news summary should be a single ${newsFormat}.
    - **Summary Length:** Each summary should be approximately ${wordLength} words.
    - **Additional Instructions:** ${additionalInstructions || 'None'}

    **Instructions:**
    1.  **Main Title:** Create a compelling title for the newsletter that reflects the key themes.
    2.  **Introduction:** Write a short, engaging introduction (2-3 sentences) that sets the stage.
    3.  **Article Sections:** For each article, create a section with:
        - A clear, bolded heading (<h4>) with the publication date next to it in a smaller, subtle format (e.g., using a <small> tag).
        - A summary of the article, adhering to the specified format (paragraph/bullets) and length (~${wordLength} words).
        - A "Read More" link (<a>) to the original article, using the source name as the link text.
    4.  **Closing:** Add a brief closing remark.
    5.  **Styling:** Generate clean, modern HTML. Prioritize semantic tags like <h1>, <h2>, <p>, <h4>, <a>, <strong>, <em>, <ul>, <li>, and <hr>. You may use tasteful inline CSS for styling, but do not include <style> blocks.
    6.  **Output:** Provide a single block of HTML code, starting with <h1>.

    **News Articles Data (with publication dates):**
    ${articlesJson}
    `;

    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
        });
        
        let newsletterHtml = response.text.trim();
        if (!newsletterHtml) {
            throw new Error("The AI returned an empty response for the newsletter.");
        }

        // Strip markdown fences if present
        const markdownMatch = newsletterHtml.match(/```(?:html)?\s*([\s\S]*?)\s*```/);
        if (markdownMatch && markdownMatch[1]) {
            newsletterHtml = markdownMatch[1].trim();
        }

        return newsletterHtml;

    } catch (error) {
        console.error("Error generating newsletter with Gemini:", error);
        throw new Error("Failed to generate the newsletter from the AI. Please try again.");
    }
};

export const generateHeaderImage = async (prompt: string): Promise<string> => {
  try {
    const response = await ai.models.generateImages({
      model: 'imagen-4.0-generate-001',
      prompt: prompt,
      config: {
        numberOfImages: 1,
        outputMimeType: 'image/png',
        aspectRatio: '16:9',
      },
    });

    if (!response.generatedImages || response.generatedImages.length === 0) {
      throw new Error('The AI returned no images.');
    }

    const base64ImageBytes: string = response.generatedImages[0].image.imageBytes;
    return base64ImageBytes;
  } catch (error) {
    console.error("Error generating header image with Gemini:", error);
    throw new Error("Failed to generate the header image from the AI. Please try again.");
  }
};

export const refineNewsletter = async (
  currentHtml: string,
  articles: NewsArticle[],
  refinementPrompt: string
): Promise<string> => {
  const prompt = `You are an AI assistant helping a user refine a newsletter. You will be given the current newsletter HTML, the original news articles it was based on, and a user's request for changes. Your task is to process the request and provide an updated newsletter.

**User Request:** "${refinementPrompt}"

**Current Newsletter HTML:**
${currentHtml}

**Original News Articles (for context, including publication dates):**
${JSON.stringify(articles, null, 2)}

**Instructions:**
1.  **Analyze the User Request:** Understand what the user wants to change. This could be content, style, structure, or even the header image.
2.  **Handle Image Requests:** If the user asks for a new image (e.g., "change the image to be about space exploration"), you MUST ONLY respond with a JSON object of this exact format: \`{"requestType": "image", "newImagePrompt": "a detailed prompt for the new image based on the user request"}\`. Do not return any HTML or other text.
3.  **Handle Content/Data Requests:** If the user asks to add information like dates, use the "Original News Articles" data. For example, if asked to add dates, modify the HTML to include them next to the article headings.
4.  **Handle Style/Theme Changes:** For requests involving visual changes (e.g., "make it look more futuristic", "change the colors"), you are allowed to add or modify inline CSS styles directly on the HTML elements. Use styles to change colors, fonts, spacing, etc., to match the user's request.
5.  **Return Full HTML:** For any request that is not an image change, you must return the FULL, complete, updated HTML for the newsletter body.
6.  **Output Format:**
    -   If it's an image request, output ONLY the JSON object described in step 2.
    -   For all other requests, output ONLY the updated HTML.

Now, generate the response based on the user's request.`;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
    });
    
    let resultText = response.text.trim();
    if (!resultText) {
      throw new Error("The AI returned an empty response for the refinement request.");
    }

    // The AI might return JSON for an image request, or HTML for content changes.
    // It might wrap either in markdown fences. Let's strip them first.
    const markdownMatch = resultText.match(/```(?:json|html)?\s*([\s\S]*?)\s*```/);
    if (markdownMatch && markdownMatch[1]) {
        resultText = markdownMatch[1].trim();
    }

    return resultText;
  } catch (error) {
    console.error("Error refining newsletter with Gemini:", error);
    throw new Error("Failed to refine the newsletter from the AI. Please try again.");
  }
};