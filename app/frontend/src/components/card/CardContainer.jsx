import React from "react";
import { Container, Paper } from "@mui/material";

const CardContainer = ({ children, sx={} }) => {
  return (
    <Container
      component={Paper}
      elevation={6}
      sx={{
        p: 4,
        width: "100%",
        // minWidth: {
        //   xs:350, 
        //   sm: 500, 
        //   md: 700, 
        //   lg: 900, 
        // },
        maxWidth: {
          xs:350, 
          sm: 500, 
          md: 700, 
          lg: 800, 
        },
        backgroundColor: "rgba(255, 255, 255, 0.9)",
        borderRadius: "12px",
        textAlign: "center",
        ...sx,
      }}
    >
      {children}
    </Container>
  );
};

export default CardContainer;
