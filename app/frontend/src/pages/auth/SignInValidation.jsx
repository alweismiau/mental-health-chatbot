const Validation = (user) => {
    let errors = {};
    const email_pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const password_pattern = /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$/;
  
    if (!user.email.trim()) {
      errors.email = "Wajib memasukkan Email";
    } else if (!email_pattern.test(user.email)) {
      errors.email = "Email tidak Valid";
    }
  
    if (!user.password.trim()) {
      errors.password = "Wajib memasukkan Password";
    } else if (!password_pattern.test(user.password)) {
      errors.password = "Password harus terdiri dari 8 karakter, termasuk 1 huruf dan 1 nomor";
    }
  
    return errors;
  };
  
  export default Validation;
  