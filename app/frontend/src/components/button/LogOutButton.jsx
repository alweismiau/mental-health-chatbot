import React, { useState, useEffect } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { Box, IconButton } from "@mui/material";
import LogoutIcon from "@mui/icons-material/Logout";

const LogOutButton = () => {
  const navigate = useNavigate();

  const handleSignOut = () => {
    localStorage.removeItem("token");
    navigate("/signin");
  };
  return (
    <Box
      sx={{
        position: "absolute",
        top: {
          xs: 10,
          lg: 20,
        },
        right: 30,
      }}
    >
      <IconButton onClick={handleSignOut} sx={{ color: "#7BB5E8" }}>
        <LogoutIcon fontSize="large" />
      </IconButton>
    </Box>
  );
};

export default LogOutButton;
