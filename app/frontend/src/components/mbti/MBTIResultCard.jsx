import React from "react";
import { Typography, Box, Paper, Grid } from "@mui/material";
import { useNavigate } from "react-router-dom";
import CustomButton from "../button/CustomButton";

const MBTIResultCard = ({ result, description, images }) => {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        p:{xs: 2, lg: 4},
        borderRadius: 2,
        maxWidth: 1200,
        mx: "auto",
        bgcolor: "#ffffff",
      }}
    >
      <Grid container spacing={4} alignItems="center">
        <Grid item xs={12} md={5}>
          <Box sx={{ textAlign: { xs: "center", md: "left" } }}>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              Kamu adalah seorang
            </Typography>
            <Typography
              variant="h3"
              sx={{
                fontWeight: "bold",
                color: "#7BB5E8",
                my: 1,
              }}
            >
              {result}
            </Typography>
            <Typography
              variant="body1"
              sx={{
                lineHeight: 1.6,
                color: "text.secondary",
                textAlign: { xs: "center", md: "justify" },
              }}
            >
              {description || "Deskripsi tidak tersedia untuk tipe ini."}
            </Typography>
            <Box
              sx={{
                display: { xs: "flex", md: "block" },
                justifyContent: "center",
                mt: 3,
              }}
            >
              <CustomButton
                text="Masuk ke chatbot"
                onClick={() => navigate("/dashboard/chatbot")}
              />
            </Box>
          </Box>
        </Grid>

        <Grid item xs={12} md={7}>
          <Box
            sx={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
              gap: 3,
            }}
          >
            {images && (
              <Box
                component="img"
                src={images}
                alt={`Gambar MBTI ${result}`}
                sx={{
                  width: "100%",
                  borderRadius: 2,
                  boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                }}
              />
            )}
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MBTIResultCard;
