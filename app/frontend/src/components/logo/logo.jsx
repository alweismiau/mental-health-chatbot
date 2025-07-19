import React from "react";
import { Box } from "@mui/material";
import LogoImage from "../../assets/logo/logo-blue.png";

const logoVariants = {
  center: {
    position: "fixed",
    top: {
      xs: 10,
      lg: 20,
    },
    left: "50%",
    transform: {
      xs: "translateX(-50%)",
      sm: "translateX(-50%)",
      md: "translateX(-50%)",
    },
    width: {
      // xs: 60,
      // sm: 80,
      xs: 100,
      md: 140,
    },
    height: "auto",
    zIndex: 1000,
  },
  left: {
    position: "fixed",
    top: {
      xs: 10,
      lg: 20,
    },
    left: {
      xs: "50%",
      sm: "50%",
      md: "50%",
      lg: 130,
    },
    transform: {
      xs: "translateX(-50%)",
      md: "translateX(-50%)",
    },
    width: {
      xs: 60,
      sm: 80,
      md: 100,
      lg: 140,
    },
    height: "auto",
    zIndex: 1000,
  },
};

const Logo = ({ variant = "center" }) => {
  return (
    <Box
      component="img"
      src={LogoImage}
      alt="Logo"
      sx={logoVariants[variant]}
    />
  );
};

export default Logo;
