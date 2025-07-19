const prisma = require("../db");

// Mencari user berdasarkan email
const findUserByEmail = async (email) => {
  return await prisma.user.findUnique({ where: { email } });
};

// Membuat user baru
const createUser = async (name, email, hashedPassword) => {
  return await prisma.user.create({
    data: { name, email, password: hashedPassword },
  });
};

module.exports = { findUserByEmail, createUser };
