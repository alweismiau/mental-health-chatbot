import React, { useState, useEffect } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { fetchUserProfile, isAuthenticated } from "../../auth";
import { Box, Typography } from "@mui/material";
import CustomContainer from "../../components/container/CustomContainer";
import CardContainer from "../../components/card/CardContainer";
import CustomButton from "../../components/button/CustomButton";
import LogOutButton from "../../components/button/LogOutButton";
import Chatbot from "../chatbot/Chatbot";

const Dashboard = () => {
  const [user, setUser] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    async function fetchProfile() {
      if (!isAuthenticated()) {
        console.log("User not authenticated, redirecting to Sign In");
        navigate("/signin");
        return;
      }

      const getUserById = await fetchUserProfile();
      console.log("User Profile Data:", getUserById);
      if (!getUserById) {
        navigate("/signin");
      } else {
        setUser(getUserById);
      }
    }

    fetchProfile();
  }, [navigate]);

  if (location.pathname === "/dashboard/mbti-test") {
    return <Outlet />;
  }

  if (location.pathname === "/dashboard/chatbot") {
    return <Chatbot />;
  }

  if (!user) return <p>Loading...</p>;

  const handleMBTITest = () => {
    navigate("/dashboard/mbti-test");
  };

  const handleChatbot = () => {
    navigate("/dashboard/chatbot");
  };

  return (
    <CustomContainer sx={{ flexDirection: "column" }}>
      <LogOutButton />
      <Box
        sx={{
          textAlign: "center",
          mb: 2,
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
        }}
      >
        <Typography variant="h4" color="#212529">
          Selamat Datang
        </Typography>
        <Typography
          variant="h4"
          fontWeight="bold"
          sx={{ color: "#7BB5E8", ml: 1 }}
        >
          {user.user.name.charAt(0).toUpperCase() + user.user.name.slice(1)}

        </Typography>
        <Typography variant="h5" color="#212529">
          !👋
        </Typography>
      </Box>
      <CardContainer>
        {/* <Typography
          variant="h6"
          fontWeight="bold"
          sx={{
            maxWidth: {xs: "95%", sm: "70%", md: "80%"},
            mx: "auto",
          }}
        >
          Sebelum masuk ke fitur chatbot, yuk <b>Tes MBTI</b> kamu!
        </Typography> */}
        <Typography
          variant="subtitle1"
          sx={{
            color: "grey.600",
            textAlign: "center",
            maxWidth: {xs: "95%", sm: "70%", md: "80%", lg: "90%"},
            mt: 1,
            mb: 3,
            mx: "auto",

          }}
        >
          Sebelum masuk ke fitur chatbot, yuk 
           <Box component="span" sx={{ color:"grey.900" }}> Tes MBTI</Box> kamu terlebih dulu!
          Tujuan dari test MBTI ini adalah agar chatbot mampu memberikan respons
          sesuai dengan tipe kepribadian kamu
        </Typography>

        <Box sx={{ display: "flex", justifyContent: "center", gap: 2 }}>
          <CustomButton text="MBTI TEST" onClick={handleMBTITest} />
          <CustomButton
            variant="outlined"
            text="Chatbot"
            onClick={handleChatbot}
            disabled={!user.user.mbtiResult}
          />
        </Box>
      </CardContainer>
      <Outlet />
    </CustomContainer>
  );
};

export default Dashboard;
