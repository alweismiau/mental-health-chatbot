import { createBrowserRouter, Navigate } from "react-router-dom";
import SignIn from "../pages/auth/SignIn";
import SignUp from "../pages/auth/SignUp";
import Dashboard from "../pages/dashboard/Dashboard";
import TestMBTI from "../pages/mbti/MBTI";
import Chatbot from "../pages/chatbot/Chatbot";
import { isAuthenticated } from "../auth";

const hasTakenMBTITest = () => {
  const mbtiResult = localStorage.getItem("mbtiResult");
  return mbtiResult !== null; 
};

const PrivateRoute = ({ element, requireMBTITest }) => {
  if (!isAuthenticated()) {
    return <Navigate to="/signin" />;
  }

  if (requireMBTITest && !hasTakenMBTITest()) {
    return <Navigate to="/dashboard/mbti-test" />; 
  }

  return element;
};

export const Router = createBrowserRouter([
  { path: "/signup", element: <SignUp /> },
  { path: "/signin", element: <SignIn /> },
  {
    path: "/dashboard",
    element: <PrivateRoute element={<Dashboard />} />,
    children: [
      { path: "mbti-test", element: <PrivateRoute element={<TestMBTI />} /> },
      { path: "chatbot", element: <PrivateRoute element={<Chatbot />} /> },
    ],
  },
  { path: "*", element: <Navigate to="/signin" replace /> },
]);
