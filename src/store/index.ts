import { create } from 'zustand';
import type { Difficulty, Style, Mode, MCQ, Flashcard } from '@/types';

// App State
interface AppState {
  // Current view
  currentView: 'home' | 'results' | 'library' | 'practice';
  setCurrentView: (view: 'home' | 'results' | 'library' | 'practice') => void;

  // Generation settings
  mode: Mode;
  difficulty: Difficulty;
  style: Style;
  questionCount: number;
  setMode: (mode: Mode) => void;
  setDifficulty: (difficulty: Difficulty) => void;
  setStyle: (style: Style) => void;
  setQuestionCount: (count: number) => void;

  // Generated content
  currentDocumentId: string | null;
  mcqs: MCQ[];
  flashcards: Flashcard[];
  setGeneratedContent: (documentId: string, mcqs: MCQ[], flashcards: Flashcard[]) => void;
  clearGeneratedContent: () => void;

  // Loading state
  isGenerating: boolean;
  setIsGenerating: (isGenerating: boolean) => void;

  // Practice mode
  practiceMode: 'mcq' | 'flashcard' | null;
  currentQuestionIndex: number;
  answers: Record<string, string>;
  practiceResults: Record<string, boolean>;
  setPracticeMode: (mode: 'mcq' | 'flashcard' | null) => void;
  setCurrentQuestionIndex: (index: number) => void;
  setAnswer: (questionId: string, answer: string) => void;
  setPracticeResult: (questionId: string, isCorrect: boolean) => void;
  resetPractice: () => void;

  // Theme
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Current view
  currentView: 'home',
  setCurrentView: (view) => set({ currentView: view }),

  // Generation settings
  mode: 'mcq',
  difficulty: 'medium',
  style: 'high_yield',
  questionCount: 5,
  setMode: (mode) => set({ mode }),
  setDifficulty: (difficulty) => set({ difficulty }),
  setStyle: (style) => set({ style }),
  setQuestionCount: (count) => set({ questionCount: count }),

  // Generated content
  currentDocumentId: null,
  mcqs: [],
  flashcards: [],
  setGeneratedContent: (documentId, mcqs, flashcards) =>
    set({ currentDocumentId: documentId, mcqs, flashcards }),
  clearGeneratedContent: () =>
    set({ currentDocumentId: null, mcqs: [], flashcards: [] }),

  // Loading state
  isGenerating: false,
  setIsGenerating: (isGenerating) => set({ isGenerating }),

  // Practice mode
  practiceMode: null,
  currentQuestionIndex: 0,
  answers: {},
  practiceResults: {},
  setPracticeMode: (mode) => set({ practiceMode: mode, currentQuestionIndex: 0 }),
  setCurrentQuestionIndex: (index) => set({ currentQuestionIndex: index }),
  setAnswer: (questionId, answer) =>
    set((state) => ({
      answers: { ...state.answers, [questionId]: answer },
    })),
  setPracticeResult: (questionId, isCorrect) =>
    set((state) => ({
      practiceResults: { ...state.practiceResults, [questionId]: isCorrect },
    })),
  resetPractice: () =>
    set({
      practiceMode: null,
      currentQuestionIndex: 0,
      answers: {},
      practiceResults: {},
    }),

  // Theme
  theme: 'light',
  toggleTheme: () =>
    set((state) => ({
      theme: state.theme === 'light' ? 'dark' : 'light',
    })),
}));
