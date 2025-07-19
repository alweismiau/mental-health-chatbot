import axios from "axios";

const API_URL = "https://brave-wired-mastiff.ngrok-free.app";

const api = axios.create({
  baseURL: API_URL,
  // headers: {
  //   "Content-Type": "application/json",
  // },
});

export default api;
