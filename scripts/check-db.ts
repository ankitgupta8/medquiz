import { config } from "dotenv";
import { resolve } from "path";

// Load .env file explicitly
config({ path: resolve(process.cwd(), ".env") });

import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

async function main() {
  console.log("=== Database Status ===\n");

  // Check users
  const users = await prisma.user.findMany({
    select: {
      id: true,
      email: true,
      name: true,
      createdAt: true,
      _count: {
        select: {
          documents: true,
          bookmarks: true,
        },
      },
    },
  });

  console.log("Users:", users.length);
  for (const u of users) {
    console.log(
      `  - ${u.email} (${u.name || "no name"}) - ${u._count.documents} docs, ${u._count.bookmarks} bookmarks`
    );
  }

  // Check documents
  const documents = await prisma.document.findMany({
    select: {
      id: true,
      title: true,
      mode: true,
      createdAt: true,
      user: { select: { email: true } },
      _count: { select: { mcqs: true, flashcards: true } },
    },
    orderBy: { createdAt: "desc" },
    take: 5,
  });

  console.log("\nRecent Documents:", documents.length);
  for (const d of documents) {
    console.log(
      `  - ${d.title || "Untitled"} (${d.mode}) by ${d.user.email} - ${d._count.mcqs} MCQs, ${d._count.flashcards} flashcards`
    );
  }

  // Total counts
  const totalMcqs = await prisma.mCQ.count();
  const totalFlashcards = await prisma.flashcard.count();
  const totalBookmarks = await prisma.bookmark.count();

  console.log("\n=== Summary ===");
  console.log(`Total MCQs: ${totalMcqs}`);
  console.log(`Total Flashcards: ${totalFlashcards}`);
  console.log(`Total Bookmarks: ${totalBookmarks}`);
}

main()
  .catch((e) => {
    console.error("Error:", e.message);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
