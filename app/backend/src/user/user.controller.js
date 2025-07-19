const prisma = require("../db");

const getUserById = async (req, res) => {

try {
  const userId = req.user.id; 

  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { id: true, name: true, email: true, mbtiResult: true },
  });

  if (!user) {
    return res.status(404).json({ message: "User not found" });
  }

  res.status(200).json({ user });
} catch (error) {
  res.status(500).json({ message: "Server error", error: error.message });
}
};

const getAllUsers = async (req, res) => {
  try {
    const users = await prisma.user.findMany({
      select: { id: true, name: true, email: true, mbtiResult: true },
    });

    res.status(200).json({
      count: users.length,
      users,
    });
  } catch (error) {
    res.status(500).json({ message: "Server error", error });
  }
};

const updateMBTIResult = async (req, res) => {
  try {
    const userId = req.user.id; 
    const { mbtiResult } = req.body; 

    if (!mbtiResult || mbtiResult.length !== 4) {
      return res.status(400).json({ message: "Invalid MBTI result" });
    }

    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: { mbtiResult },
      select: { id: true, name: true, email: true, mbtiResult: true },
    });

    res.status(200).json({ message: "MBTI result updated successfully", user: updatedUser });
  } catch (error) {
    res.status(500).json({ message: "Server error", error: error.message });
  }
};

module.exports = { getUserById, getAllUsers, updateMBTIResult};
