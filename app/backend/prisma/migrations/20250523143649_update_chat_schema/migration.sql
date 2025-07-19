/*
  Warnings:

  - The primary key for the `ChatHistory` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to drop the column `chatId` on the `ChatHistory` table. All the data in the column will be lost.
  - You are about to drop the column `message` on the `ChatHistory` table. All the data in the column will be lost.
  - You are about to drop the column `sender` on the `ChatHistory` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "ChatHistory" DROP CONSTRAINT "ChatHistory_pkey",
DROP COLUMN "chatId",
DROP COLUMN "message",
DROP COLUMN "sender",
ALTER COLUMN "id" DROP DEFAULT,
ALTER COLUMN "id" SET DATA TYPE TEXT,
ADD CONSTRAINT "ChatHistory_pkey" PRIMARY KEY ("id");
DROP SEQUENCE "ChatHistory_id_seq";

-- CreateTable
CREATE TABLE "ChatMessage" (
    "id" TEXT NOT NULL,
    "chatId" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "sender" TEXT NOT NULL,
    "emotion" TEXT,
    "responseTime" DOUBLE PRECISION,

    CONSTRAINT "ChatMessage_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "ChatMessage" ADD CONSTRAINT "ChatMessage_chatId_fkey" FOREIGN KEY ("chatId") REFERENCES "ChatHistory"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
