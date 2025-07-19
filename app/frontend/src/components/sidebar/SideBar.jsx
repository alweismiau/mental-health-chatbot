import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { fetchUserProfile, isAuthenticated } from "../../auth";
import CustomButton from "../../components/button/CustomButton";
import { Box, Typography, IconButton } from "@mui/material";
import DriveFileRenameOutlineIcon from "@mui/icons-material/DriveFileRenameOutline";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import DeleteIcon from "@mui/icons-material/Delete";
import axios from "axios";
import "../../pages/chatbot/Chatbot.css";
const EXPRESS_URL = "https://brave-wired-mastiff.ngrok-free.app";

const SideBar = ({
  startNewChat,
  loadChatHistory,
  deleteChatHistory,
  deleteAllHistories,
  setChatHistories,
}) => {
  const [user, setUser] = useState(null);
  const [fetchedHistories, setFetchedHistories] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    async function fetchUserAndChatHistories() {
      if (!isAuthenticated()) {
        navigate("/signin");
        return;
      }

      try {
        const userData = await fetchUserProfile();
        if (!userData || !userData.user?.id) {
          navigate("/signin");
          return;
        }

        setUser(userData);
        const userId = userData.user.id;

        const response = await axios.get(`${EXPRESS_URL}/get-chats/${userId}`, {
          headers: { "ngrok-skip-browser-warning": "69420" },
        });
        const data = Array.isArray(response.data) ? response.data : [];

        setFetchedHistories(data);
        setChatHistories(data);
      } catch (error) {
        console.error("❌ Gagal mengambil data user atau riwayat:", error);
      }
    }

    fetchUserAndChatHistories();
  }, [navigate, setChatHistories]);

  if (!user) return <p>Loading...</p>;

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        borderRight: "2px solid #ccc",
        justifyContent: "space-between",
        height: "100vh",
        width: { xs: "40vw", md: "20vw", lg: "15vw" },
      }}
      py={2}
      px={4}
    >
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          mt: { xs: 4, lg: 16 },
          gap: 2,
          alignItems: "left",
        }}
      >
        <Box sx={{ display: "flex", flexDirection: "column", mt: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Nama User:
          </Typography>
          <Typography variant="h5" color="text.primary" sx={{ mt: 1 }}>
            {user?.user?.name?.charAt(0).toUpperCase() +
              user?.user?.name?.slice(1)}
          </Typography>
        </Box>

        <Box sx={{ display: "flex", flexDirection: "column" }}>
          <Typography variant="body2" color="text.secondary">
            Tipe MBTI:
          </Typography>
          <Typography variant="h5" color="text.primary" sx={{ mt: 1 }}>
            {user.user.mbtiResult}
          </Typography>
        </Box>

        <Box sx={{ display: "flex", flexDirection: "column" }}>
          <Typography variant="body2" color="text.secondary">
            Riwayat
          </Typography>

          <Box
            className="chat-container"
            sx={{
              width: "100%",
              maxHeight: "40vh",
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 1,
              pr: 1,
              mt: 2,
            }}
          >
            {fetchedHistories.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                Tidak ada riwayat chat.
              </Typography>
            ) : (
              fetchedHistories.map((history, index) => (
                <Box
                  key={history.id}
                  sx={{ display: "flex", alignItems: "center", mb: 1 }}
                >
                  <CustomButton
                    variant="text"
                    startIcon={<ChatBubbleOutlineIcon />}
                    text={`Riwayat ${index + 1}`}
                    onClick={() => loadChatHistory(history.id)}
                    sx={{ fontSize: "14px", flexGrow: 1 }}
                  />

                  <IconButton
                    onClick={() => deleteChatHistory(history.id)}
                    sx={{ color: "red", "&:hover": { color: "#EE4E4E" } }}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
              ))
            )}
          </Box>
        </Box>
      </Box>

      <Box sx={{ mb: 1 }}>
        <CustomButton
          startIcon={<DriveFileRenameOutlineIcon />}
          text="Chat Baru"
          onClick={startNewChat}
        />
      </Box>
    </Box>
  );
};

export default SideBar;
