import React, { useState } from "react";
import { TextField, InputAdornment, IconButton } from "@mui/material";
import { Visibility, VisibilityOff } from "@mui/icons-material";

const FormFields = ({ fields, values, handleChange, errors }) => {
  const [showPassword, setShowPassword] = useState(false);

  const handleTogglePassword = () => {
    setShowPassword((prev) => !prev);
  };

  return (
    <>
      {fields.map((field) => (
        <TextField
          key={field.name}
          label={field.label}
          name={field.name}
          type={field.type === "password" && !showPassword ? "password" : "text"}
          fullWidth
          variant="filled" 
          margin="normal"
          value={values[field.name]}
          onChange={handleChange}
          error={!!errors[field.name]}
          helperText={errors[field.name] || (field.name === "password" ? "Password harus terdiri dari 8 karakter, termasuk 1 huruf dan 1 nomor!" : "")}
          InputProps={{
            disableUnderline: true, 
            sx: {
              backgroundColor: "#F5F7FA",
              borderRadius: "8px",
              padding: "2px",
              "&:hover": { backgroundColor: "#EDEFF2" },
              "&.Mui-focused": { backgroundColor: "#EDEFF2" },
            },
            ...(field.type === "password" && {
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton onClick={handleTogglePassword} >
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }),
          }}
          InputLabelProps={{
            shrink: true, 
            sx: { color: "#6C757D", fontSize: "14px", fontWeight: 600 },
          }}
          required
        />
      ))}
    </>
  );
};

export default FormFields;
