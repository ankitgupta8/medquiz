import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server-client';
import { db } from '@/lib/db';
import { 
  generateMCQsFromChunks, 
  generateFlashcardsFromChunks,
  withRetry,
  AIModel 
} from '@/services/ai-service';
import { getProcessingStats } from '@/lib/text-processing';
import type { Difficulty, Style, Mode } from '@/types';

export async function POST(request: NextRequest) {
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

    const body = await request.json();
    const { 
      text, 
      mode, 
      difficulty, 
      style, 
      title, 
      questionCount = 5,
      maxChunkSize = 10000,
      model = 'gpt-oss'
    } = body as {
      text: string;
      mode: Mode;
      difficulty: Difficulty;
      style: Style;
      title?: string;
      questionCount?: number;
      maxChunkSize?: number;
      model?: AIModel;
    };

    // Validate input
    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      return NextResponse.json(
        { success: false, error: 'Text is required and cannot be empty' },
        { status: 400 }
      );
    }

    if (!['mcq', 'flashcard'].includes(mode)) {
      return NextResponse.json(
        { success: false, error: 'Mode must be "mcq" or "flashcard"' },
        { status: 400 }
      );
    }

    if (!['easy', 'medium', 'hard'].includes(difficulty)) {
      return NextResponse.json(
        { success: false, error: 'Difficulty must be "easy", "medium", or "hard"' },
        { status: 400 }
      );
    }

    if (!['clinical', 'high_yield', 'exam'].includes(style)) {
      return NextResponse.json(
        { success: false, error: 'Style must be "clinical", "high_yield", or "exam"' },
        { status: 400 }
      );
    }

    // Validate question count
    const count = Math.max(1, questionCount);

    // Get processing stats
    const stats = getProcessingStats(text, maxChunkSize, count);
    
    console.log('Processing stats:', {
      totalChars: stats.totalChars,
      chunkCount: stats.chunkCount,
      avgChunkSize: stats.avgChunkSize,
      distribution: stats.questionsPerChunk,
    });

    // Create document record linked to user
    const document = await db.document.create({
      data: {
        userId,
        originalText: text.trim(),
        mode,
        difficulty,
        style,
        title: title || `Generated ${mode === 'mcq' ? 'MCQs' : 'Flashcards'} (${count} items)`,
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

    // Generate content using AI with chunking and parallel processing
    try {
      if (mode === 'mcq') {
        const result = await withRetry(() => 
          generateMCQsFromChunks(text, difficulty, style, count, maxChunkSize, undefined, model)
        );
        mcqs = result.mcqs;
        
        console.log(`Generated ${mcqs.length} MCQs from ${result.chunks.length} chunks using model: ${model}`);

        // Save MCQs to database
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
          generateFlashcardsFromChunks(text, difficulty, style, count, maxChunkSize, undefined, model)
        );
        flashcards = result.flashcards;
        
        console.log(`Generated ${flashcards.length} flashcards from ${result.chunks.length} chunks`);

        // Save flashcards to database
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
      console.error('Generation error:', error);
      // Clean up the document if generation failed
      await db.document.delete({ where: { id: document.id } });
      throw error;
    }

    // Fetch the complete document with relations
    const completeDocument = await db.document.findUnique({
      where: { id: document.id },
      include: {
        mcqs: true,
        flashcards: true,
      },
    });

    return NextResponse.json({
      success: true,
      documentId: document.id,
      mcqs: completeDocument?.mcqs || [],
      flashcards: completeDocument?.flashcards || [],
      stats: {
        totalChunks: stats.chunkCount,
        requestedCount: count,
        generatedCount: mcqs.length || flashcards.length,
      },
    });
  } catch (error) {
    console.error('Generation error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to generate content',
      },
      { status: 500 }
    );
  }
}

// GET endpoint to preview processing stats
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const textLength = parseInt(searchParams.get('textLength') || '0');
  const questionCount = parseInt(searchParams.get('questionCount') || '5');
  const maxChunkSize = parseInt(searchParams.get('maxChunkSize') || '10000');

  if (textLength === 0) {
    return NextResponse.json({
      success: true,
      stats: {
        chunkCount: 0,
        estimatedTime: '0 seconds',
      },
    });
  }

  const chunkCount = Math.ceil(textLength / maxChunkSize);
  const avgPerChunk = Math.round(questionCount / Math.max(1, chunkCount));
  const estimatedSeconds = chunkCount * 8; // ~8 seconds per chunk

  return NextResponse.json({
    success: true,
    stats: {
      totalChars: textLength,
      chunkCount,
      avgPerChunk,
      estimatedTime: estimatedSeconds < 60 
        ? `${estimatedSeconds} seconds` 
        : `${Math.round(estimatedSeconds / 60)} minutes`,
    },
  });
}
