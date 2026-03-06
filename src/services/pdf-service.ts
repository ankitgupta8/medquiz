import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Types
export interface PDFPage {
  pageNumber: number;
  imagePath?: string;
  base64?: string;
  width: number;
  height: number;
}

export interface PDFProcessingResult {
  totalPages: number;
  pages: PDFPage[];
  text: string;
}

/**
 * Extract text from a PDF using pdftotext (poppler-utils)
 */
export async function extractTextFromPDF(pdfPath: string): Promise<string> {
  try {
    const { stdout } = await execAsync(`pdftotext -layout "${pdfPath}" -`);
    return stdout || '';
  } catch (error) {
    console.error('Error extracting text from PDF:', error);
    return '';
  }
}

/**
 * Get the number of pages in a PDF using pdfinfo
 */
export async function getPageCount(pdfPath: string): Promise<number> {
  try {
    const { stdout } = await execAsync(`pdfinfo "${pdfPath}"`);
    const match = stdout.match(/Pages:\s+(\d+)/);
    return match ? parseInt(match[1]) : 0;
  } catch (error) {
    console.error('Error getting page count:', error);
    return 0;
  }
}

/**
 * Check if PDF has extractable text or needs OCR
 * Returns true if the PDF appears to be scanned (needs OCR)
 */
export async function needsOCR(pdfPath: string): Promise<boolean> {
  const text = await extractTextFromPDF(pdfPath);
  const pageCount = await getPageCount(pdfPath);
  
  // If less than 50 characters extracted per page on average, likely needs OCR
  const avgCharsPerPage = text.length / Math.max(1, pageCount);
  
  console.log(`Text analysis: ${text.length} chars, ${pageCount} pages, avg ${avgCharsPerPage.toFixed(1)} chars/page`);
  
  return avgCharsPerPage < 50;
}

/**
 * Convert PDF pages to images using pdftoppm (poppler-utils)
 */
export async function convertPDFToImages(
  pdfPath: string,
  outputDir: string,
  dpi: number = 200
): Promise<string[]> {
  try {
    // Create output directory if it doesn't exist
    await fs.mkdir(outputDir, { recursive: true });
    
    // Convert PDF to images using pdftoppm
    const outputPath = path.join(outputDir, 'page');
    await execAsync(`pdftoppm -png -r ${dpi} "${pdfPath}" "${outputPath}"`);
    
    // Get list of generated images
    const files = await fs.readdir(outputDir);
    const imageFiles = files
      .filter(f => f.endsWith('.png'))
      .sort((a, b) => {
        const numA = parseInt(a.match(/\d+/)?.[0] || '0');
        const numB = parseInt(b.match(/\d+/)?.[0] || '0');
        return numA - numB;
      })
      .map(f => path.join(outputDir, f));
    
    return imageFiles;
  } catch (error) {
    console.error('Error converting PDF to images:', error);
    throw new Error('Failed to convert PDF to images. Make sure poppler-utils is installed.');
  }
}

/**
 * Process a PDF file for OCR
 * Extracts text and converts pages to images if needed
 */
export async function processPDFForOCR(
  pdfPath: string,
  outputDir: string
): Promise<PDFProcessingResult> {
  const pageCount = await getPageCount(pdfPath);
  const text = await extractTextFromPDF(pdfPath);
  
  // Convert to images
  const imagePaths = await convertPDFToImages(pdfPath, outputDir);
  
  // Read images as base64
  const pages: PDFPage[] = [];
  for (let i = 0; i < imagePaths.length; i++) {
    const imagePath = imagePaths[i];
    const imageBuffer = await fs.readFile(imagePath);
    
    pages.push({
      pageNumber: i + 1,
      imagePath,
      base64: imageBuffer.toString('base64'),
      width: 0,
      height: 0,
    });
  }
  
  return {
    totalPages: pageCount,
    pages,
    text,
  };
}

/**
 * Clean up temporary files
 */
export async function cleanupTempFiles(dir: string): Promise<void> {
  try {
    await fs.rm(dir, { recursive: true, force: true });
  } catch (error) {
    console.error('Error cleaning up temp files:', error);
  }
}
