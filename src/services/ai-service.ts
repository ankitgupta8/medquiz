import OpenAI from 'openai';
import type { Difficulty, Style, MCQInput, FlashcardInput } from '@/types';
import { splitTextIntoChunks, distributeQuestionCount, type TextChunk } from '@/lib/text-processing';

// Initialize OpenAI client with Baseten configuration
const client = new OpenAI({
  apiKey: 'g1SjmcWL.vaFVDRnYiiGpYsRyjxuapGEOa6S7Gl2N',
  baseURL: 'https://inference.baseten.co/v1',
});

// AI Model type
export type AIModel = 'gpt-oss' | 'deepseek' | 'glm' | 'kimi';

// Model configurations
const MODEL_CONFIGS: Record<AIModel, { model: string; name: string }> = {
  'gpt-oss': {
    model: 'openai/gpt-oss-120b',
    name: 'GPT OSS 120B',
  },
  'deepseek': {
    model: 'deepseek-ai/DeepSeek-V3-0324',
    name: 'DeepSeek V3 0324',
  },
  'glm': {
    model: 'zai-org/GLM-4.7',
    name: 'GLM 4.7',
  },
  'kimi': {
    model: 'moonshotai/Kimi-K2-Thinking',
    name: 'Kimi K2',
  },
};

const DEFAULT_MODEL: AIModel = 'gpt-oss';

// Progress callback type
export type ProgressCallback = (processed: number, total: number, chunkIndex: number) => void;

// Generate MCQs from text
export async function generateMCQs(
  text: string,
  difficulty: Difficulty,
  style: Style,
  count: number = 5,
  model: AIModel = DEFAULT_MODEL
): Promise<MCQInput[]> {
  const difficultyInstructions = {
    easy: 'Create recall-based questions that test basic knowledge and definitions. Questions should be straightforward with clear correct answers.',
    medium: 'Create questions that require concept integration and understanding of relationships. Questions should connect multiple concepts.',
    hard: 'Create application-based questions requiring multi-step reasoning. Include complex scenarios that require analysis.',
  };

  const styleInstructions = {
    clinical: 'Include patient scenarios, clinical presentations, and case-based questions. Use medical terminology appropriately.',
    high_yield: 'Focus on direct facts, key definitions, and essential concepts. Make questions clear and concise.',
    exam: 'Design questions with strategic distractors that test pattern recognition. Include common misconceptions as wrong answers.',
  };

  const systemPrompt = `You are an expert medical educator creating high-quality multiple-choice questions for academic learning.

DIFFICULTY LEVEL: ${difficulty}
${difficultyInstructions[difficulty]}

QUESTION STYLE: ${style}
${styleInstructions[style]}

You must generate exactly ${count} questions in valid JSON format. Each question must have:
- A clear question stem
- Exactly 4 options (A, B, C, D)
- Exactly one correct answer
- A helpful explanation
- A short hint to guide learning

CRITICAL JSON FORMATTING RULES:
1. Return ONLY a valid JSON array - no markdown code blocks, no extra text
2. All string values MUST be on a SINGLE LINE - do NOT include actual newlines in strings
3. Use \\n for line breaks within strings (escaped newlines)
4. Do NOT use quotes inside string values - use single quotes instead if needed
5. Every opening quote must have a closing quote on the same line

Example of CORRECT format:
[{"question":"What is X?","options":{"A":"Option A","B":"Option B","C":"Option C","D":"Option D"},"correctAnswer":"A","explanation":"A is correct because...","hint":"Think about...","difficulty":"${difficulty}","style":"${style}"}]`;

  const userPrompt = `Based on the following academic text, generate ${count} multiple-choice questions.

IMPORTANT: Return a single-line JSON array. Each question object on ONE line. Use \\n for line breaks within text.

Text:
---
${text}
---

Return JSON array (each object on one line, use \\n for newlines inside strings):
[{"question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"correctAnswer":"A","explanation":"...","hint":"...","difficulty":"${difficulty}","style":"${style}"},...]`;

  const modelConfig = MODEL_CONFIGS[model];
  console.log(`Using model: ${modelConfig.name} (${modelConfig.model})`);

  try {
    const completion = await client.chat.completions.create({
      model: modelConfig.model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      max_tokens: 16000,
      temperature: 0.7,
      top_p: 1,
      stream: false,
    });

    const response = completion.choices[0]?.message?.content;
    if (!response) {
      throw new Error('Empty response from AI');
    }

    // Parse and validate JSON with safe parser
    const mcqs: MCQInput[] = parseJsonSafe<MCQInput[]>(response, 'mcq');

    // Validate each MCQ and filter out invalid ones
    const validMcqs: MCQInput[] = [];
    for (const mcq of mcqs) {
      try {
        validateMCQ(mcq);
        validMcqs.push(mcq);
      } catch (validationError) {
        console.warn('Skipping invalid MCQ:', validationError);
      }
    }

    return validMcqs;
  } catch (error) {
    console.error('Error generating MCQs:', error);
    throw new Error(`Failed to generate MCQs: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

// Generate Flashcards from text
export async function generateFlashcards(
  text: string,
  difficulty: Difficulty,
  style: Style,
  count: number = 5,
  model: AIModel = DEFAULT_MODEL
): Promise<FlashcardInput[]> {
  const difficultyInstructions = {
    easy: 'Create definition recall cards with straightforward Q&A format. Focus on key terms and basic concepts.',
    medium: 'Create concept explanation cards that require understanding. Include relationships and mechanisms.',
    hard: 'Create applied understanding cards with clinical correlations. Test ability to apply knowledge.',
  };

  const styleInstructions = {
    clinical: 'Include clinical correlations, patient presentations, and practical applications. Use case-based learning.',
    high_yield: 'Focus on essential facts and high-yield concepts for exam preparation. Be concise and clear.',
    exam: 'Create cards that help with pattern recognition and common exam topics. Include test-taking tips.',
  };

  const systemPrompt = `You are an expert medical educator creating high-quality flashcards for academic learning.

DIFFICULTY LEVEL: ${difficulty}
${difficultyInstructions[difficulty]}

CARD STYLE: ${style}
${styleInstructions[style]}

You must generate exactly ${count} flashcards in valid JSON format. Each flashcard must have:
- A front side (question, concept, or prompt) - keep this concise
- A back side (answer, explanation, or definition) - MUST use rich markdown formatting

MARKDOWN FORMATTING for the "back" field (use these extensively):
- Use **bold** for key terms, important concepts, and emphasis
- Use *italic* for secondary emphasis
- Use bullet points: - Item 1\\n- Item 2\\n- Item 3
- Use numbered lists: 1. First\\n2. Second\\n3. Third
- Use > for clinical pearls or important notes
- Use ### for subheadings if needed
- Use code format \`code\` for technical terms if applicable

CRITICAL JSON FORMATTING RULES:
1. Return ONLY a valid JSON array - no markdown code blocks around the JSON
2. Use \\n for line breaks within strings (escaped newlines, NOT actual newlines)
3. Use \\" for any quotes inside strings (escaped quotes)
4. Each flashcard object can span multiple lines in the JSON for readability
5. Make the back field RICH with formatting - don't just use plain text!

Example with rich formatting:
[
  {
    "front": "What are the key features of Condition X?",
    "back": "**Condition X** is characterized by:\\n\\n- **Key feature 1**: Description here\\n- **Key feature 2**: Another important point\\n- **Key feature 3**: Third point\\n\\n> Clinical pearl: Remember to check for associated findings!",
    "difficulty": "${difficulty}",
    "style": "${style}"
  }
]

Notice how the back field uses **bold**, bullet points with \\n, and blockquotes.`;

  const userPrompt = `Based on the following academic text, generate ${count} flashcards with RICH MARKDOWN formatting.

Use **bold** for key terms, bullet points for lists, and > for important notes.

Text:
---
${text}
---

Return JSON array. Use \\n for line breaks inside strings. Make the back field richly formatted:
[
  {
    "front": "Question or concept",
    "back": "**Answer** with formatting:\\n\\n- **Point 1**: Description\\n- **Point 2**: Description\\n\\n> Key takeaway",
    "difficulty": "${difficulty}",
    "style": "${style}"
  }
]`;

  const modelConfig = MODEL_CONFIGS[model];
  console.log(`Using model: ${modelConfig.name} (${modelConfig.model})`);

  try {
    const completion = await client.chat.completions.create({
      model: modelConfig.model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      max_tokens: 16000,
      temperature: 0.7,
      top_p: 1,
      stream: false,
    });

    const response = completion.choices[0]?.message?.content;
    if (!response) {
      throw new Error('Empty response from AI');
    }

    // Parse and validate JSON with safe parser
    const flashcards: FlashcardInput[] = parseJsonSafe<FlashcardInput[]>(response, 'flashcard');

    // Validate each flashcard and filter out invalid ones
    const validFlashcards: FlashcardInput[] = [];
    for (const card of flashcards) {
      try {
        validateFlashcard(card);
        validFlashcards.push(card);
      } catch (validationError) {
        console.warn('Skipping invalid flashcard:', validationError);
      }
    }

    return validFlashcards;
  } catch (error) {
    console.error('Error generating flashcards:', error);
    throw new Error(`Failed to generate flashcards: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Generate MCQs from multiple chunks in parallel
 * This is the main function for handling large texts
 */
export async function generateMCQsFromChunks(
  text: string,
  difficulty: Difficulty,
  style: Style,
  totalCount: number,
  maxChunkSize: number = 10000,
  onProgress?: ProgressCallback,
  model: AIModel = DEFAULT_MODEL
): Promise<{ mcqs: MCQInput[]; chunks: TextChunk[]; distribution: number[] }> {
  // Split text into chunks
  const chunks = splitTextIntoChunks(text, maxChunkSize);
  
  // Distribute question count across chunks
  const distribution = distributeQuestionCount(chunks, totalCount);
  
  console.log(`Processing ${chunks.length} chunks with distribution:`, distribution);
  
  // Process all chunks in parallel
  const promises = chunks.map(async (chunk, index) => {
    const count = distribution[index];
    if (count <= 0) return [];
    
    try {
      const result = await generateMCQs(chunk.text, difficulty, style, count, model);
      
      // Report progress
      if (onProgress) {
        onProgress(1, chunks.length, index);
      }
      
      return result;
    } catch (error) {
      console.error(`Error processing chunk ${index}:`, error);
      // Return empty array for failed chunks
      return [];
    }
  });
  
  // Wait for all chunks to complete
  const results = await Promise.all(promises);
  
  // Flatten all MCQs
  const allMcqs = results.flat();
  
  return { mcqs: allMcqs, chunks, distribution };
}

/**
 * Generate Flashcards from multiple chunks in parallel
 */
export async function generateFlashcardsFromChunks(
  text: string,
  difficulty: Difficulty,
  style: Style,
  totalCount: number,
  maxChunkSize: number = 10000,
  onProgress?: ProgressCallback,
  model: AIModel = DEFAULT_MODEL
): Promise<{ flashcards: FlashcardInput[]; chunks: TextChunk[]; distribution: number[] }> {
  // Split text into chunks
  const chunks = splitTextIntoChunks(text, maxChunkSize);
  
  // Distribute question count across chunks
  const distribution = distributeQuestionCount(chunks, totalCount);
  
  console.log(`Processing ${chunks.length} chunks with distribution:`, distribution);
  
  // Process all chunks in parallel
  const promises = chunks.map(async (chunk, index) => {
    const count = distribution[index];
    if (count <= 0) return [];
    
    try {
      const result = await generateFlashcards(chunk.text, difficulty, style, count, model);
      
      // Report progress
      if (onProgress) {
        onProgress(1, chunks.length, index);
      }
      
      return result;
    } catch (error) {
      console.error(`Error processing chunk ${index}:`, error);
      return [];
    }
  });
  
  // Wait for all chunks to complete
  const results = await Promise.all(promises);
  
  // Flatten all flashcards
  const allFlashcards = results.flat();
  
  return { flashcards: allFlashcards, chunks, distribution };
}

// Clean JSON response by removing markdown code blocks and extracting JSON
function cleanJsonResponse(response: string): string {
  let cleaned = response.trim();
  
  // Check if response looks like HTML
  if (cleaned.startsWith('<!DOCTYPE') || cleaned.startsWith('<html') || cleaned.includes('<body')) {
    console.error('Received HTML response instead of JSON:', cleaned.substring(0, 500));
    throw new Error('AI returned an HTML error page instead of JSON. This might be an API issue. Please try again.');
  }
  
  // Remove various markdown code block formats
  // Handle ```json ... ```
  if (cleaned.includes('```json')) {
    const startIndex = cleaned.indexOf('```json') + 7;
    const endIndex = cleaned.indexOf('```', startIndex);
    if (endIndex > startIndex) {
      cleaned = cleaned.slice(startIndex, endIndex);
    }
  }
  // Handle ``` ... ```
  else if (cleaned.includes('```')) {
    const startIndex = cleaned.indexOf('```') + 3;
    // Skip language identifier if present (e.g., ```json, ```JavaScript)
    let jsonStart = startIndex;
    while (jsonStart < cleaned.length && cleaned[jsonStart] !== '\n' && cleaned[jsonStart] !== '[' && cleaned[jsonStart] !== '{') {
      jsonStart++;
    }
    if (cleaned[jsonStart] === '\n') jsonStart++;
    
    const endIndex = cleaned.indexOf('```', jsonStart);
    if (endIndex > jsonStart) {
      cleaned = cleaned.slice(jsonStart, endIndex);
    }
  }
  
  cleaned = cleaned.trim();
  
  // Try to extract JSON array if there's extra content
  const arrayStart = cleaned.indexOf('[');
  const arrayEnd = cleaned.lastIndexOf(']');
  
  if (arrayStart !== -1 && arrayEnd !== -1 && arrayEnd > arrayStart) {
    cleaned = cleaned.slice(arrayStart, arrayEnd + 1);
  }
  
  return cleaned.trim();
}

// Default values for difficulty and style
const DEFAULT_DIFFICULTY: Difficulty = 'medium';
const DEFAULT_STYLE: Style = 'high_yield';
const VALID_DIFFICULTIES: Difficulty[] = ['easy', 'medium', 'hard'];
const VALID_STYLES: Style[] = ['clinical', 'high_yield', 'exam'];

// Normalize and validate difficulty/style values
function normalizeDifficulty(value: unknown): Difficulty {
  if (typeof value === 'string' && VALID_DIFFICULTIES.includes(value as Difficulty)) {
    return value as Difficulty;
  }
  return DEFAULT_DIFFICULTY;
}

function normalizeStyle(value: unknown): Style {
  if (typeof value === 'string' && VALID_STYLES.includes(value as Style)) {
    return value as Style;
  }
  return DEFAULT_STYLE;
}

// Extract valid JSON objects from malformed response
function extractValidObjects(jsonString: string, type: 'flashcard' | 'mcq'): object[] {
  const objects: object[] = [];
  
  // Try to find all complete JSON objects
  let depth = 0;
  let startIndex = -1;
  let inString = false;
  let escapeNext = false;
  
  for (let i = 0; i < jsonString.length; i++) {
    const char = jsonString[i];
    
    if (escapeNext) {
      escapeNext = false;
      continue;
    }
    
    if (char === '\\') {
      escapeNext = true;
      continue;
    }
    
    if (char === '"') {
      inString = !inString;
      continue;
    }
    
    if (!inString) {
      if (char === '{') {
        if (depth === 0) {
          startIndex = i;
        }
        depth++;
      } else if (char === '}') {
        depth--;
        if (depth === 0 && startIndex !== -1) {
          const objStr = jsonString.slice(startIndex, i + 1);
          try {
            const obj = JSON.parse(objStr);
            // Validate it's the right type and has all required fields
            if (type === 'flashcard' && obj.front && obj.back) {
              // Ensure difficulty and style are valid
              obj.difficulty = normalizeDifficulty(obj.difficulty);
              obj.style = normalizeStyle(obj.style);
              objects.push(obj);
            } else if (type === 'mcq' && obj.question && obj.options && obj.correctAnswer && obj.explanation) {
              // Ensure difficulty and style are valid
              obj.difficulty = normalizeDifficulty(obj.difficulty);
              obj.style = normalizeStyle(obj.style);
              // Ensure hint exists
              if (!obj.hint || typeof obj.hint !== 'string') {
                obj.hint = 'Think about the key concepts in the question.';
              }
              objects.push(obj);
            }
          } catch {
            // Skip invalid objects
          }
          startIndex = -1;
        }
      }
    }
  }
  
  return objects;
}

// Try to repair common JSON issues
function repairJson(jsonString: string, type: 'flashcard' | 'mcq' = 'flashcard'): string {
  let repaired = jsonString;
  
  // First, try parsing as-is
  try {
    JSON.parse(repaired);
    return repaired;
  } catch {
    // Continue with repair attempts
  }
  
  // Remove any trailing commas before ] or }
  repaired = repaired.replace(/,(\s*[\]\}])/g, '$1');
  
  // Try parsing after removing trailing commas
  try {
    JSON.parse(repaired);
    return repaired;
  } catch {
    // Continue
  }
  
  // Try to find first valid JSON array by finding matching brackets
  let arrayStart = repaired.indexOf('[');
  while (arrayStart !== -1) {
    let depth = 0;
    let foundEnd = -1;
    
    for (let i = arrayStart; i < repaired.length; i++) {
      if (repaired[i] === '[') depth++;
      else if (repaired[i] === ']') {
        depth--;
        if (depth === 0) {
          foundEnd = i;
          break;
        }
      }
    }
    
    if (foundEnd > arrayStart) {
      const candidate = repaired.slice(arrayStart, foundEnd + 1);
      try {
        JSON.parse(candidate);
        return candidate;
      } catch {
        // Try next array
      }
    }
    
    arrayStart = repaired.indexOf('[', arrayStart + 1);
  }
  
  // Try to extract individual valid objects and reconstruct array
  const validObjects = extractValidObjects(repaired, type);
  if (validObjects.length > 0) {
    const reconstructed = JSON.stringify(validObjects);
    console.log(`Extracted ${validObjects.length} valid objects from malformed response`);
    return reconstructed;
  }
  
  // Return original if repair failed
  return jsonString;
}

// Parse JSON with better error handling
function parseJsonSafe<T>(response: string, type: 'flashcard' | 'mcq' = 'flashcard'): T {
  const cleaned = cleanJsonResponse(response);
  
  try {
    return JSON.parse(cleaned) as T;
  } catch (error) {
    console.error('Failed to parse JSON initially. Attempting repair...');
    console.error('Parse error:', error);
    
    // Try to repair the JSON
    const repaired = repairJson(cleaned, type);
    
    try {
      const result = JSON.parse(repaired) as T;
      console.log('Successfully parsed repaired JSON');
      return result;
    } catch (repairError) {
      console.error('Failed to parse JSON after repair. Response preview:', cleaned.substring(0, 1000));
      console.error('Repair error:', repairError);
      
      // Try to provide more helpful error message
      if (cleaned.includes('<!DOCTYPE') || cleaned.includes('<html')) {
        throw new Error('AI returned an error page. Please try again.');
      }
      
      if (cleaned.length === 0) {
        throw new Error('AI returned empty response. Please try again.');
      }
      
      throw new Error(`Failed to parse AI response as JSON. The response may be malformed. Preview: ${cleaned.substring(0, 200)}...`);
    }
  }
}

// Validate MCQ structure
function validateMCQ(mcq: unknown): asserts mcq is MCQInput {
  if (typeof mcq !== 'object' || mcq === null) {
    throw new Error('Invalid MCQ: not an object');
  }

  const m = mcq as Record<string, unknown>;

  if (typeof m.question !== 'string' || m.question.trim().length === 0) {
    throw new Error('Invalid MCQ: missing or empty question');
  }

  if (typeof m.options !== 'object' || m.options === null) {
    throw new Error('Invalid MCQ: missing options object');
  }

  const options = m.options as Record<string, unknown>;
  if (!['A', 'B', 'C', 'D'].every((key) => typeof options[key] === 'string')) {
    throw new Error('Invalid MCQ: options must have A, B, C, D as strings');
  }

  if (!['A', 'B', 'C', 'D'].includes(m.correctAnswer as string)) {
    throw new Error('Invalid MCQ: correctAnswer must be A, B, C, or D');
  }

  if (typeof m.explanation !== 'string' || m.explanation.trim().length === 0) {
    throw new Error('Invalid MCQ: missing or empty explanation');
  }

  // Ensure hint exists
  if (!m.hint || typeof m.hint !== 'string') {
    m.hint = 'Think about the key concepts in the question.';
  }

  // Ensure difficulty and style have valid values
  m.difficulty = normalizeDifficulty(m.difficulty);
  m.style = normalizeStyle(m.style);
}

// Validate Flashcard structure
function validateFlashcard(card: unknown): asserts card is FlashcardInput {
  if (typeof card !== 'object' || card === null) {
    throw new Error('Invalid Flashcard: not an object');
  }

  const c = card as Record<string, unknown>;

  if (typeof c.front !== 'string' || c.front.trim().length === 0) {
    throw new Error('Invalid Flashcard: missing or empty front');
  }

  if (typeof c.back !== 'string' || c.back.trim().length === 0) {
    throw new Error('Invalid Flashcard: missing or empty back');
  }

  // Ensure difficulty and style have valid values
  c.difficulty = normalizeDifficulty(c.difficulty);
  c.style = normalizeStyle(c.style);
}

// Retry wrapper for AI calls
export async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3
): Promise<T> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown error');
      console.error(`Attempt ${attempt} failed:`, lastError.message);

      if (attempt < maxRetries) {
        // Exponential backoff
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
      }
    }
  }

  throw lastError || new Error('All retry attempts failed');
}
