// Type definitions for MCQ and Flashcard application

export type Difficulty = 'easy' | 'medium' | 'hard';
export type Style = 'clinical' | 'high_yield' | 'exam';
export type Mode = 'mcq' | 'flashcard';

// MCQ Types
export interface MCQOption {
  A: string;
  B: string;
  C: string;
  D: string;
}

export interface MCQ {
  id: string;
  question: string;
  options: MCQOption;
  correctAnswer: 'A' | 'B' | 'C' | 'D';
  explanation: string;
  hint?: string;
  difficulty: Difficulty;
  style: Style;
}

export interface MCQInput {
  question: string;
  options: MCQOption;
  correctAnswer: string;
  explanation: string;
  hint?: string;
  difficulty: string;
  style: string;
}

// Flashcard Types
export interface Flashcard {
  id: string;
  front: string;
  back: string;
  mnemonic?: string;
  clinicalCorrelation?: string;
  keyPoint?: string;
  difficulty: Difficulty;
  style: Style;
}

export interface FlashcardInput {
  front: string;
  back: string;
  mnemonic?: string;
  clinicalCorrelation?: string;
  keyPoint?: string;
  difficulty: string;
  style: string;
}

// Document Types
export interface Document {
  id: string;
  title?: string;
  originalText: string;
  mode: Mode;
  difficulty: Difficulty;
  style: Style;
  createdAt: Date;
  updatedAt: Date;
  mcqs?: MCQ[];
  flashcards?: Flashcard[];
}

// Generation Request Types
export interface GenerateRequest {
  text: string;
  mode: Mode;
  difficulty: Difficulty;
  style: Style;
  title?: string;
  questionCount?: number;
}

export interface GenerateResponse {
  success: boolean;
  documentId?: string;
  mcqs?: MCQ[];
  flashcards?: Flashcard[];
  error?: string;
}

// Practice Session Types
export interface PracticeAnswer {
  itemId: string;
  isCorrect: boolean;
  userAnswer: string;
}

export interface PracticeSessionResult {
  totalItems: number;
  correctCount: number;
  incorrectCount: number;
  percentage: number;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Filter Types
export interface FilterOptions {
  mode?: Mode;
  difficulty?: Difficulty;
  style?: Style;
  search?: string;
}

// Difficulty and Style Labels
export const DIFFICULTY_LABELS: Record<Difficulty, string> = {
  easy: 'Easy - Recall-based',
  medium: 'Medium - Concept integration',
  hard: 'Hard - Application & reasoning',
};

export const STYLE_LABELS: Record<Style, string> = {
  clinical: 'Clinical-Based - Patient scenarios',
  high_yield: 'High-Yield - Direct facts',
  exam: 'Exam-Oriented - Pattern recognition',
};

export const DIFFICULTY_DESCRIPTIONS: Record<Difficulty, string> = {
  easy: 'Recall-based questions testing basic knowledge',
  medium: 'Concept integration requiring understanding of relationships',
  hard: 'Application and multi-step reasoning problems',
};

export const STYLE_DESCRIPTIONS: Record<Style, string> = {
  clinical: 'Include patient scenarios and clinical presentations',
  high_yield: 'Direct fact-based questions for quick review',
  exam: 'Pattern recognition with strategic distractors',
};
