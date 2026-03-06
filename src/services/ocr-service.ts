import ZAI from 'z-ai-web-dev-sdk';
import type { PDFPage } from './pdf-service';

// Types
export interface OCRResult {
  pageNumber: number;
  text: string;
  success: boolean;
  error?: string;
}

export interface BatchOCRResult {
  results: OCRResult[];
  totalPages: number;
  successfulPages: number;
  failedPages: number;
  combinedText: string;
}

// Initialize ZAI SDK
let zaiInstance: Awaited<ReturnType<typeof ZAI.create>> | null = null;

async function getZAI() {
  if (!zaiInstance) {
    zaiInstance = await ZAI.create();
  }
  return zaiInstance;
}

/**
 * Perform OCR on a single page image using VLM
 */
export async function ocrPage(
  page: PDFPage,
  context?: string
): Promise<OCRResult> {
  try {
    if (!page.base64) {
      return {
        pageNumber: page.pageNumber,
        text: '',
        success: false,
        error: 'No image data available',
      };
    }
    
    const zai = await getZAI();
    
    const prompt = `You are an OCR system specialized in extracting text from academic and educational documents.

Extract ALL text from this image with high accuracy. Follow these guidelines:
1. Preserve the document structure (headings, paragraphs, lists)
2. Maintain the original formatting where possible
3. Include any formulas, equations, or special characters
4. If there are tables, represent them in a structured format
5. If there are diagrams or figures, describe them briefly in [brackets]
${context ? `\nContext: This is page ${context} of a document.` : ''}

Return ONLY the extracted text, nothing else.`;

    const response = await zai.chat.completions.createVision({
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: prompt,
            },
            {
              type: 'image_url',
              image_url: {
                url: `data:image/png;base64,${page.base64}`,
              },
            },
          ],
        },
      ],
      thinking: { type: 'disabled' },
    });

    const text = response.choices[0]?.message?.content || '';
    
    return {
      pageNumber: page.pageNumber,
      text,
      success: true,
    };
  } catch (error) {
    console.error(`OCR failed for page ${page.pageNumber}:`, error);
    return {
      pageNumber: page.pageNumber,
      text: '',
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Perform OCR on multiple pages in parallel with batching
 */
export async function ocrPages(
  pages: PDFPage[],
  batchSize: number = 3,
  onProgress?: (current: number, total: number) => void
): Promise<BatchOCRResult> {
  const results: OCRResult[] = [];
  let completed = 0;
  
  // Process in batches to avoid rate limiting
  for (let i = 0; i < pages.length; i += batchSize) {
    const batch = pages.slice(i, i + batchSize);
    
    // Process batch in parallel
    const batchResults = await Promise.all(
      batch.map((page) => ocrPage(page))
    );
    
    results.push(...batchResults);
    completed += batch.length;
    
    if (onProgress) {
      onProgress(completed, pages.length);
    }
    
    // Small delay between batches to avoid rate limiting
    if (i + batchSize < pages.length) {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  
  const successfulPages = results.filter((r) => r.success).length;
  const failedPages = results.filter((r) => !r.success).length;
  
  // Combine text in page order
  const combinedText = results
    .sort((a, b) => a.pageNumber - b.pageNumber)
    .map((r) => r.text)
    .join('\n\n---\n\n');
  
  return {
    results,
    totalPages: pages.length,
    successfulPages,
    failedPages,
    combinedText,
  };
}
