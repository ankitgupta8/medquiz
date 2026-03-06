/**
 * Text Processing Utilities
 * Handles chunking and processing of large OCR texts
 */

export interface TextChunk {
  index: number;
  text: string;
  startChar: number;
  endChar: number;
}

/**
 * Split text into chunks of approximately the specified size
 * Tries to split at sentence boundaries for better context preservation
 */
export function splitTextIntoChunks(text: string, maxChunkSize: number = 10000): TextChunk[] {
  if (text.length <= maxChunkSize) {
    return [{
      index: 0,
      text: text.trim(),
      startChar: 0,
      endChar: text.length,
    }];
  }

  const chunks: TextChunk[] = [];
  let currentIndex = 0;
  let chunkIndex = 0;

  while (currentIndex < text.length) {
    let endPosition = Math.min(currentIndex + maxChunkSize, text.length);

    // If not at the end of text, try to find a good break point
    if (endPosition < text.length) {
      // Look for sentence boundary (period, question mark, exclamation followed by space or newline)
      const searchText = text.slice(currentIndex, endPosition);
      
      // Try to find the last sentence boundary
      const sentenceBreakers = ['. ', '.\n', '? ', '?\n', '! ', '!\n', '\n\n'];
      let lastBreakIndex = -1;

      for (const breaker of sentenceBreakers) {
        const lastIndex = searchText.lastIndexOf(breaker);
        if (lastIndex > lastBreakIndex) {
          lastBreakIndex = lastIndex + breaker.length;
        }
      }

      // If found a good break point within reasonable range, use it
      // Otherwise, try to break at a space
      if (lastBreakIndex > maxChunkSize * 0.5) {
        endPosition = currentIndex + lastBreakIndex;
      } else {
        // Try to break at a space
        const lastSpaceIndex = searchText.lastIndexOf(' ');
        if (lastSpaceIndex > maxChunkSize * 0.5) {
          endPosition = currentIndex + lastSpaceIndex + 1;
        }
      }
    }

    const chunkText = text.slice(currentIndex, endPosition).trim();
    
    if (chunkText.length > 0) {
      chunks.push({
        index: chunkIndex,
        text: chunkText,
        startChar: currentIndex,
        endChar: endPosition,
      });
      chunkIndex++;
    }

    currentIndex = endPosition;
  }

  return chunks;
}

/**
 * Calculate question distribution across chunks
 * Distributes questions proportionally based on chunk size
 */
export function distributeQuestionCount(
  chunks: TextChunk[],
  totalQuestions: number
): number[] {
  if (chunks.length === 0) return [];
  if (chunks.length === 1) return [totalQuestions];

  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.text.length, 0);
  const distribution: number[] = [];
  let assigned = 0;

  // Calculate proportional distribution
  for (let i = 0; i < chunks.length; i++) {
    const proportion = chunks[i].text.length / totalLength;
    let count = Math.round(totalQuestions * proportion);
    
    // Ensure at least 1 question per chunk if totalQuestions >= chunkCount
    if (totalQuestions >= chunks.length && count < 1) {
      count = 1;
    }
    
    distribution.push(count);
    assigned += count;
  }

  // Adjust for rounding errors
  const difference = totalQuestions - assigned;
  if (difference !== 0) {
    // Distribute the difference across chunks
    const step = difference > 0 ? 1 : -1;
    let remaining = Math.abs(difference);
    let idx = 0;
    
    while (remaining > 0) {
      distribution[idx % distribution.length] += step;
      remaining--;
      idx++;
    }
  }

  return distribution;
}

/**
 * Get processing statistics
 */
export function getProcessingStats(
  text: string,
  maxChunkSize: number,
  totalQuestions: number
): {
  totalChars: number;
  chunkCount: number;
  avgChunkSize: number;
  questionsPerChunk: number[];
  estimatedTimeMs: number;
} {
  const chunks = splitTextIntoChunks(text, maxChunkSize);
  const distribution = distributeQuestionCount(chunks, totalQuestions);
  
  // Estimate processing time (roughly 5-10 seconds per chunk)
  const estimatedTimeMs = chunks.length * 7000;

  return {
    totalChars: text.length,
    chunkCount: chunks.length,
    avgChunkSize: Math.round(text.length / chunks.length),
    questionsPerChunk: distribution,
    estimatedTimeMs,
  };
}
