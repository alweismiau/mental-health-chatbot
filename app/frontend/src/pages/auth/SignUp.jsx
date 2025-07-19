import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Snackbar, Alert } from "@mui/material";
import Validation from "./SignUpValidation";
import axios from "../../api/api";
import CustomContainer from "../../components/container/CustomContainer";
import FormHead from "../../components/auth/FormHead";
import FormFields from "../../components/auth/FormFields";
import CardContainer from "../../components/card/CardContainer";
import CustomButton from "../../components/button/CustomButton";

const Signup = () => {
  const [user, setUser] = useState({
    name: "",
    email: "",
    password: "",
  });

  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState("");
  const navigate = useNavigate();

  const handleInput = (event) => {
    setUser((prev) => ({
      ...prev,
      [event.target.name]: event.target.value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const validationErrors = Validation(user);
    setErrors(validationErrors);
    setServerError("");

    if (Object.keys(validationErrors).length === 0) {
      setLoading(true);
      try {
        const response = await axios.post("/signup", user);
        // alert(response.data.message);
        // navigate("/signin");
        setSuccessMessage(response.data.message || "Pendaftaran berhasil!");
        setOpenSnackbar(true);
        setTimeout(() => {
          navigate("/signin");
        }, 2000);
      } catch (error) {
        setServerError(error.response?.data?.message || "Something went wrong");
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <CustomContainer>
      <CardContainer>
        <Snackbar
          open={openSnackbar}
          autoHideDuration={2000}
          onClose={() => setOpenSnackbar(false)}
          anchorOrigin={{ vertical: "top", horizontal: "center" }}
        >
          <Alert
            onClose={() => setOpenSnackbar(false)}
            severity="success"
            sx={{ width: "100%" }}
          >
            {successMessage}
          </Alert>
        </Snackbar>

        <FormHead
          variant="sign-up"
          title="Buat Akun"
          href="/signin"
          sx={{ mb: 4, textAlign: "center" }}
        />

        {serverError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {serverError}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <FormFields
            fields={[
              { name: "name", label: "Nama Panjang" },
              { name: "email", label: "Alamat Email", type: "email" },
              { name: "password", label: "Password", type: "password" },
            ]}
            values={user}
            handleChange={handleInput}
            errors={errors}
          />
          <CustomButton
            sx={{ mt: 2 }}
            type="submit"
            text={loading ? "Sedang daftar..." : "Daftar"}
            disabled={loading}
          />
        </form>
      </CardContainer>
    </CustomContainer>
  );
};

export default Signup;
