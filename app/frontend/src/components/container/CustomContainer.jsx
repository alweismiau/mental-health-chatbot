import React from "react";
import { Box } from "@mui/material";
import Logo from "../logo/Logo";
import { Link } from "react-router-dom";

const CustomContainer = ({ children, sx = {}, variant = "center" }) => {
  return (
    <Box
      sx={{
        display: "flex",
        width: "100vw",
        height: "100vh",
        alignItems: "center",
        justifyContent: "center",
        ...sx,
      }}
    >
      <Link to="/dashboard" style={{ textDecoration: "none" }}>
        <Logo variant={variant} />
      </Link>
      {children}
    </Box>
  );
};

export default CustomContainer;
