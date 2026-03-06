import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server-client';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs/promises';
import { v4 as uuidv4 } from 'uuid';
import { 
  generateMCQsFromChunks, 
  generateFlashcardsFromChunks,
  withRetry,
  AIModel 
} from '@/services/ai-service';
import { db } from '@/lib/db';
import type { Difficulty, Style, Mode } from '@/types';

const execAsync = promisify(exec);

// Pages per chunk for PDF processing
const PAGES_PER_CHUNK = 30;

// Max chunks to process in parallel
const MAX_CONCURRENT_CHUNKS = 2;

// Timeout for API calls (5 minutes)
const API_TIMEOUT = 5 * 60 * 1000;

// Temporary directory for PDF processing
const TEMP_DIR = '/tmp/pdf-processing';

// Mistral API Key - from environment
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY || '';
const MISTRAL_BASE_URL = 'https://api.mistral.ai/v1';

/**
 * Sleep utility
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Fetch with timeout
 */
async function fetchWithTimeout(url: string, options: RequestInit, timeout: number): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Get the number of pages in a PDF using pdfinfo
 */
async function getPDFPageCount(pdfPath: string): Promise<number> {
  try {
    const { stdout } = await execAsync(`pdfinfo "${pdfPath}"`);
    const match = stdout.match(/Pages:\s+(\d+)/);
    return match ? parseInt(match[1]) : 0;
  } catch (error) {
    console.error('[PDF] Error getting page count:', error);
    return 0;
  }
}

/**
 * Split PDF into chunks using qpdf
 */
async function splitPDFIntoChunks(pdfPath: string, outputDir: string, pagesPerChunk: number): Promise<string[]> {
  const totalPages = await getPDFPageCount(pdfPath);
  const chunks: string[] = [];
  
  console.log(`[PDF] Splitting ${totalPages} pages into ${pagesPerChunk}-page chunks`);
  
  for (let startPage = 1; startPage <= totalPages; startPage += pagesPerChunk) {
    const endPage = Math.min(startPage + pagesPerChunk - 1, totalPages);
    const chunkPath = path.join(outputDir, `chunk_${startPage}_${endPage}.pdf`);
    
    try {
      await execAsync(`qpdf --empty --pages "${pdfPath}" ${startPage}-${endPage} -- "${chunkPath}"`);
      chunks.push(chunkPath);
      console.log(`[PDF] Created chunk: pages ${startPage}-${endPage}`);
    } catch {
      try {
        await execAsync(`gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dFirstPage=${startPage} -dLastPage=${endPage} -sOutputFile="${chunkPath}" "${pdfPath}"`);
        chunks.push(chunkPath);
        console.log(`[PDF] Created chunk with gs: pages ${startPage}-${endPage}`);
      } catch (error) {
        console.error(`[PDF] Failed to create chunk pages ${startPage}-${endPage}`);
      }
    }
  }
  
  return chunks;
}

/**
 * Upload file to Mistral and get signed URL with retry
 */
async function uploadToMistralAndGetUrl(filePath: string, retries: number = 3): Promise<string> {
  let lastError: Error | null = null;
  
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const fileBuffer = await fs.readFile(filePath);
      const fileName = path.basename(filePath);
      
      // Create form data
      const formData = new FormData();
      const blob = new Blob([fileBuffer], { type: 'application/pdf' });
      formData.append('file', blob, fileName);
      formData.append('purpose', 'ocr');
      
      // Upload file with timeout
      const uploadResponse = await fetchWithTimeout(
        `${MISTRAL_BASE_URL}/files`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${MISTRAL_API_KEY}`,
          },
          body: formData,
        },
        API_TIMEOUT
      );
      
      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text();
        throw new Error(`Upload failed: ${uploadResponse.status} - ${errorText}`);
      }
      
      const uploadResult = await uploadResponse.json();
      const fileId = uploadResult.id;
      console.log(`[Mistral] Uploaded file, ID: ${fileId}`);
      
      // Get signed URL with timeout
      const signedUrlResponse = await fetchWithTimeout(
        `${MISTRAL_BASE_URL}/files/${fileId}/url?expiry=24`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${MISTRAL_API_KEY}`,
            'Accept': 'application/json',
          },
        },
        API_TIMEOUT
      );
      
      if (!signedUrlResponse.ok) {
        const errorText = await signedUrlResponse.text();
        throw new Error(`Get signed URL failed: ${signedUrlResponse.status} - ${errorText}`);
      }
      
      const signedUrlResult = await signedUrlResponse.json();
      return signedUrlResult.url;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown error');
      console.error(`[Mistral] Upload attempt ${attempt}/${retries} failed:`, lastError.message);
      
      if (attempt < retries) {
        const delay = attempt * 2000; // Exponential backoff
        console.log(`[Mistral] Retrying in ${delay}ms...`);
        await sleep(delay);
      }
    }
  }
  
  throw lastError || new Error('Upload failed after retries');
}

/**
 * Process OCR using Mistral API with retry
 */
async function processOCR(documentUrl: string, retries: number = 3): Promise<string> {
  let lastError: Error | null = null;
  
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetchWithTimeout(
        `${MISTRAL_BASE_URL}/ocr`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${MISTRAL_API_KEY}`,
          },
          body: JSON.stringify({
            model: 'mistral-ocr-latest',
            document: {
              type: 'document_url',
              document_url: documentUrl,
            },
            include_image_base64: false,
          }),
        },
        API_TIMEOUT
      );
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`OCR failed: ${response.status} - ${errorText}`);
      }
      
      const ocrResult = await response.json();
      
      // Extract text from pages
      let extractedText = '';
      if (ocrResult.pages && Array.isArray(ocrResult.pages)) {
        for (const page of ocrResult.pages) {
          if (page.markdown) {
            extractedText += page.markdown + '\n\n--- PAGE BREAK ---\n\n';
          }
        }
      }
      
      return extractedText;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown error');
      console.error(`[OCR] Attempt ${attempt}/${retries} failed:`, lastError.message);
      
      if (attempt < retries) {
        const delay = attempt * 3000;
        console.log(`[OCR] Retrying in ${delay}ms...`);
        await sleep(delay);
      }
    }
  }
  
  throw lastError || new Error('OCR failed after retries');
}

/**
 * Process a single chunk with retry
 */
async function processChunk(chunkPath: string, index: number, totalChunks: number): Promise<string> {
  const maxRetries = 3;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`[Chunk ${index + 1}/${totalChunks}] Uploading (attempt ${attempt})...`);
      const signedUrl = await uploadToMistralAndGetUrl(chunkPath);
      
      console.log(`[Chunk ${index + 1}/${totalChunks}] Running OCR...`);
      const text = await processOCR(signedUrl);
      
      console.log(`[Chunk ${index + 1}/${totalChunks}] Done: ${text.length} chars`);
      return text;
    } catch (error) {
      console.error(`[Chunk ${index + 1}/${totalChunks}] Attempt ${attempt} failed:`, error);
      
      if (attempt < maxRetries) {
        const delay = attempt * 5000;
        console.log(`[Chunk ${index + 1}/${totalChunks}] Retrying in ${delay}ms...`);
        await sleep(delay);
      }
    }
  }
  
  console.error(`[Chunk ${index + 1}/${totalChunks}] Failed after all retries`);
  return '';
}

/**
 * Process chunks sequentially (for large files) or in limited parallel
 */
async function processAllChunks(chunkPaths: string[]): Promise<string[]> {
  const results: string[] = [];
  const totalChunks = chunkPaths.length;
  
  // For large files, process sequentially to avoid rate limiting
  if (totalChunks > 5) {
    console.log(`[Process] Large PDF (${totalChunks} chunks), processing sequentially`);
    
    for (let i = 0; i < chunkPaths.length; i++) {
      const text = await processChunk(chunkPaths[i], i, totalChunks);
      if (text.trim()) {
        results.push(text);
      }
      
      // Add delay between chunks to avoid rate limiting
      if (i < chunkPaths.length - 1) {
        console.log(`[Process] Waiting 2s before next chunk...`);
        await sleep(2000);
      }
    }
  } else {
    // For smaller files, process in limited parallel
    console.log(`[Process] Small PDF (${totalChunks} chunks), processing in parallel (${MAX_CONCURRENT_CHUNKS} at a time)`);
    
    for (let i = 0; i < chunkPaths.length; i += MAX_CONCURRENT_CHUNKS) {
      const batch = chunkPaths.slice(i, i + MAX_CONCURRENT_CHUNKS);
      console.log(`[Process] Batch ${Math.floor(i / MAX_CONCURRENT_CHUNKS) + 1}/${Math.ceil(chunkPaths.length / MAX_CONCURRENT_CHUNKS)}`);
      
      const batchResults = await Promise.all(
        batch.map((chunkPath, batchIndex) => {
          return processChunk(chunkPath, i + batchIndex, totalChunks);
        })
      );
      
      results.push(...batchResults.filter(t => t.trim()));
      
      // Delay between batches
      if (i + MAX_CONCURRENT_CHUNKS < chunkPaths.length) {
        console.log(`[Process] Waiting 3s before next batch...`);
        await sleep(3000);
      }
    }
  }
  
  return results;
}

export async function POST(request: NextRequest) {
  const sessionId = uuidv4();
  const sessionTempDir = path.join(TEMP_DIR, sessionId);
  
  console.log(`[Upload] Session: ${sessionId}`);
  
  try {
    // Check authentication with Supabase
    const supabase = await createSupabaseServerClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json(
        { success: false, error: 'Authentication required' },
        { status: 401 }
      );
    }
    
    const userId = user.id;
    
    await fs.mkdir(sessionTempDir, { recursive: true });
    
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const mode = (formData.get('mode') || 'mcq') as Mode;
    const difficulty = (formData.get('difficulty') || 'medium') as Difficulty;
    const style = (formData.get('style') || 'high_yield') as Style;
    const questionCount = parseInt(formData.get('questionCount') as string) || 10;
    const model = (formData.get('model') || 'gpt-oss') as AIModel;
    const title = formData.get('title') as string;

    if (!file) {
      await fs.rm(sessionTempDir, { recursive: true, force: true });
      return NextResponse.json({ success: false, error: 'No file provided' }, { status: 400 });
    }

    if (file.type !== 'application/pdf') {
      await fs.rm(sessionTempDir, { recursive: true, force: true });
      return NextResponse.json({ success: false, error: 'File must be a PDF' }, { status: 400 });
    }

    console.log(`[Upload] File: ${file.name}, Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`);

    // Save PDF
    const pdfPath = path.join(sessionTempDir, 'input.pdf');
    await fs.writeFile(pdfPath, Buffer.from(await file.arrayBuffer()));

    // Get page count
    const totalPages = await getPDFPageCount(pdfPath);
    console.log(`[Upload] Pages: ${totalPages}`);
    
    if (totalPages === 0) {
      await fs.rm(sessionTempDir, { recursive: true, force: true });
      return NextResponse.json({ success: false, error: 'Could not read PDF' }, { status: 400 });
    }

    // Split PDF
    const chunksDir = path.join(sessionTempDir, 'chunks');
    await fs.mkdir(chunksDir, { recursive: true });
    
    const chunkPaths = await splitPDFIntoChunks(pdfPath, chunksDir, PAGES_PER_CHUNK);
    console.log(`[Upload] Created ${chunkPaths.length} chunks`);
    
    if (chunkPaths.length === 0) {
      await fs.rm(sessionTempDir, { recursive: true, force: true });
      return NextResponse.json({ success: false, error: 'Failed to split PDF' }, { status: 500 });
    }
    
    // Process chunks
    const textResults = await processAllChunks(chunkPaths);
    console.log(`[Upload] Extracted from ${textResults.length}/${chunkPaths.length} chunks`);

    // Cleanup
    try { await fs.rm(sessionTempDir, { recursive: true, force: true }); } catch {}

    // Combine text
    const extractedText = textResults.join('\n\n');

    if (!extractedText.trim()) {
      return NextResponse.json({ success: false, error: 'No text extracted from PDF. The PDF may be image-based or corrupted.' }, { status: 400 });
    }

    console.log(`[Upload] Total text: ${extractedText.length} chars`);

    // Create document linked to user
    const document = await db.document.create({
      data: {
        userId,
        originalText: extractedText,
        mode,
        difficulty,
        style,
        title: title || file.name.replace('.pdf', ''),
      },
    });

    let mcqs: Array<{
      question: string;
      options: { A: string; B: string; C: string; D: string };
      correctAnswer: string;
      explanation: string;
      hint?: string;
      difficulty: string;
      style: string;
    }> = [];
    let flashcards: Array<{
      front: string;
      back: string;
      mnemonic?: string;
      clinicalCorrelation?: string;
      keyPoint?: string;
      difficulty: string;
      style: string;
    }> = [];

    // Generate content
    try {
      const count = Math.max(1, questionCount);
      
      if (mode === 'mcq') {
        const result = await withRetry(() => 
          generateMCQsFromChunks(extractedText, difficulty, style, count, 10000, undefined, model)
        );
        mcqs = result.mcqs;
        console.log(`[Upload] Generated ${mcqs.length} MCQs`);

        if (mcqs.length > 0) {
          await db.mCQ.createMany({
            data: mcqs.map((mcq) => ({
              documentId: document.id,
              question: mcq.question,
              optionA: mcq.options.A,
              optionB: mcq.options.B,
              optionC: mcq.options.C,
              optionD: mcq.options.D,
              correctAnswer: mcq.correctAnswer,
              explanation: mcq.explanation,
              hint: mcq.hint || 'Think about the key concepts.',
              difficulty: mcq.difficulty,
              style: mcq.style,
            })),
          });
        }
      } else {
        const result = await withRetry(() => 
          generateFlashcardsFromChunks(extractedText, difficulty, style, count, 10000, undefined, model)
        );
        flashcards = result.flashcards;
        console.log(`[Upload] Generated ${flashcards.length} flashcards`);

        if (flashcards.length > 0) {
          await db.flashcard.createMany({
            data: flashcards.map((card) => ({
              documentId: document.id,
              front: card.front,
              back: card.back,
              mnemonic: card.mnemonic,
              clinicalCorrelation: card.clinicalCorrelation,
              keyPoint: card.keyPoint,
              difficulty: card.difficulty,
              style: card.style,
            })),
          });
        }
      }
    } catch (error) {
      console.error('[Upload] Generation error:', error);
      await db.document.delete({ where: { id: document.id } });
      throw error;
    }

    const completeDocument = await db.document.findUnique({
      where: { id: document.id },
      include: { mcqs: true, flashcards: true },
    });

    console.log(`[Upload] Complete!`);
    
    return NextResponse.json({
      success: true,
      documentId: document.id,
      mcqs: completeDocument?.mcqs || [],
      flashcards: completeDocument?.flashcards || [],
      stats: {
        pdfPages: totalPages,
        pdfChunks: chunkPaths.length,
        extractedChars: extractedText.length,
        generatedCount: mcqs.length || flashcards.length,
      },
    });
  } catch (error) {
    console.error('[Upload] Error:', error);
    try { await fs.rm(sessionTempDir, { recursive: true, force: true }); } catch {}
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to process PDF',
    }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({
    success: true,
    supportedTypes: ['application/pdf'],
    pagesPerChunk: PAGES_PER_CHUNK,
    ocrModel: 'Mistral OCR (mistral-ocr-latest)',
    note: 'Large PDFs are processed sequentially to avoid rate limiting',
  });
}
