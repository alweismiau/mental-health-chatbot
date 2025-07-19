const prisma = require("../db");

const getHistoryChat = async (req, res) => {
  try {
    const { userId, chatId, text, response, emotion, responseTime } = req.body;

    let chatHistory = await prisma.chatHistory.findUnique({
      where: { id: chatId },
    });

    if (!chatHistory) {
      chatHistory = await prisma.chatHistory.create({
        data: {
          id: chatId,
          userId: parseInt(userId),
          timestamp: new Date(),
        },
      });
    }

    // User
    await prisma.chatMessage.create({
      data: {
        chatId: chatId,
        text: text,
        sender: "user",
        emotion: emotion,
        responseTime: null,
      },
    });

    // Bot
    await prisma.chatMessage.create({
      data: {
        chatId: chatId,
        text: response,
        sender: "bot",
        emotion: null,
        responseTime: responseTime,
      },
    });

    res.status(200).json({ message: "Chat saved successfully" });
  } catch (error) {
    console.error("❌ Error saving chat:", error);
    res.status(500).json({ error: error.message });
  }
};

const saveChatSummary = async (req, res) => {
  try {
    const { userId, chatId, summary } = req.body;

    if (!userId || !chatId || !summary) {
      return res.status(400).json({ error: "userId, chatId, dan summary wajib diisi" });
    }

    const chatHistory = await prisma.chatHistory.findUnique({
      where: { id: chatId }
    });

    if (!chatHistory) {
      return res.status(404).json({ error: "Chat history tidak ditemukan" });
    }

    await prisma.chatHistory.update({
      where: { id: chatId },
      data: { summary: summary }
    });

    res.status(200).json({ message: "Summary berhasil disimpan" });
  } catch (error) {
    console.error("❌ Error saving summary:", error);
    res.status(500).json({ error: error.message });
  }
};

const getChatHistoriesByUser = async (req, res) => {
  const userId = parseInt(req.params.userId);

  try {
    const chats = await prisma.chatHistory.findMany({
      where: { userId: userId },
      include: {
        messages: true,
      },
      orderBy: {
        timestamp: "desc",
      },
    });

    res.json(chats);
  } catch (err) {
    console.error("❌ Gagal ambil chat history:", err);
    res.status(500).json({ error: "Internal Server Error" });
  }
};

const deleteChat = async (req, res) => {
  const { chatId } = req.params;
  try {
    await prisma.chatMessage.deleteMany({ where: { chatId } });
    await prisma.chatHistory.delete({ where: { id: chatId } });
    res.status(200).json({ message: "Chat deleted successfully" });
  } catch (err) {
    console.error("❌ Gagal hapus chat:", err);
    res.status(500).json({ error: "Internal Server Error" });
  }
};

const deleteAllChatsByUser = async (req, res) => {
  const { userId } = req.params;
  try {
    const histories = await prisma.chatHistory.findMany({ where: { userId: parseInt(userId) } });

    const chatIds = histories.map(chat => chat.id);
    await prisma.chatMessage.deleteMany({ where: { chatId: { in: chatIds } } });
    await prisma.chatHistory.deleteMany({ where: { userId: parseInt(userId) } });

    res.status(200).json({ message: "Semua riwayat berhasil dihapus" });
  } catch (err) {
    console.error("❌ Gagal hapus semua riwayat:", err);
    res.status(500).json({ error: "Internal Server Error" });
  }
};

module.exports = {
  getHistoryChat,
  saveChatSummary,
  getChatHistoriesByUser,
  deleteChat,
  deleteAllChatsByUser
};
