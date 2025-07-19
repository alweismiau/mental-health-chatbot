import React from "react";
import { Button } from "@mui/material";

const CustomButton = ({
  text,
  onClick,
  type = "button",
  disabled = false,
  variant = "contained",
  startIcon = null,
  sx = {},
}) => {
  const styles = {
    contained: {
      backgroundColor: "#397ECB",
      color: "#fff",
      "&:hover": {
        backgroundColor: "#7BB5E8",
      },
    },
    outlined: {
      border: "2px solid #397ECB",
      color: "#397ECB",
      backgroundColor: "transparent",
      "&:hover": {
        // backgroundColor: "#e0e0e0",
        color: "#7BB5E8",
        border: "2px solid #7BB5E8",
      },
    },
    text: {
      color: "text.primary",
      backgroundColor: "transparent",
      "&:hover": {
        backgroundColor: "#e0e0e0",
        color: "#397ECB",
      },
    },
  };

  return (
    <Button
      variant={variant}
      type={type}
      fullWidth
      startIcon={startIcon}
      onClick={onClick}
      disabled={disabled}
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        textTransform: "none",
        fontWeight: "bold",
        fontSize: "1rem",
        borderRadius: "8px",
        padding: "12px 12px",
        px: 2,
        ...styles[variant],
        ...sx,
      }}
    >
      {text}
    </Button>
  );
};

export default CustomButton;
