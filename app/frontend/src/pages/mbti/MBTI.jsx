import { useState } from "react";
import axios from "axios";
import {
  Box,
  Typography,
  Radio,
  FormControl,
  FormControlLabel,
  RadioGroup,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import CustomContainer from "../../components/container/CustomContainer";
import CardContainer from "../../components/card/CardContainer";
import CustomButton from "../../components/button/CustomButton";
import LogOutButton from "../../components/button/LogOutButton";
import MBTIResultCard from "../../components/mbti/MBTIResultCard";

import {
  questionsEI,
  questionsNS,
  questionsTF,
  questionsPJ,
  mbtiDescriptions,
  MBTIImages,
} from "../../data/data";

const FLASK_URL = "https://upward-midge-verbally.ngrok-free.app";
const EXPRESS_URL = "https://brave-wired-mastiff.ngrok-free.app";
const allQuestions = [
  ...questionsEI,
  ...questionsNS,
  ...questionsTF,
  ...questionsPJ,
];

const TestMBTI = () => {
  const [session, setSession] = useState(0);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState(Array(allQuestions.length).fill(null));
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [isFinished, setIsFinished] = useState(false);

  const navigate = useNavigate();

  const handleAnswerChange = (value) => {
    const updatedAnswers = [...answers];
    updatedAnswers[currentQuestionIndex] = value;
    setAnswers(updatedAnswers);
  };

  const handleNext = () => {
    if (currentQuestionIndex < allQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      const response = await axios.post(
        `${FLASK_URL}/mbti-test`,
        {
          answers,
        }
      );

      if (response.data.mbti_result) {
        setResult(response.data.mbti_result);
        console.log("MBTI Response:", response.data);
        setIsFinished(true);

        await axios.post(
          `${EXPRESS_URL}/update-mbti`,
          { mbtiResult: response.data.mbti_result },
          {
            headers: {
              Authorization: `Bearer ${localStorage.getItem("token")}`,
            },
          }
        );

        setSession(40);
      } else {
        console.error("MBTI result is empty!");
      }
    } catch (error) {
      console.error("Error submitting MBTI test:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <CustomContainer>
      <LogOutButton />
      <CardContainer>
        {isFinished ? (
          <MBTIResultCard
            result={result}
            description={mbtiDescriptions[result]}
            images={MBTIImages[result]}
          />
        ) : (
          <>
            <Typography
              variant="subtitle1"
              fontWeight="bold"
              gutterBottom
              color="text.secondary"
            >
              Pertanyaan {currentQuestionIndex + 1} / {allQuestions.length}
            </Typography>

            <FormControl sx={{ display: "block", mb: 2 }}>
              <Typography variant="h6" sx={{ fontWeight: "bold", mb: 2 }}>
                {allQuestions[currentQuestionIndex].question}
              </Typography>
              <RadioGroup
                value={answers[currentQuestionIndex] || ""}
                onChange={(e) => handleAnswerChange(e.target.value)}
              >
                {allQuestions[currentQuestionIndex].options.map(
                  (option, index) => (
                    <FormControlLabel
                      key={index}
                      value={option.value}
                      control={<Radio />}
                      label={option.text}
                    />
                  )
                )}
              </RadioGroup>
            </FormControl>

            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                mt: 3,
                gap: 2,
              }}
            >
              {currentQuestionIndex > 0 && (
                <CustomButton text="Back" onClick={handleBack} />
              )}
              {currentQuestionIndex < allQuestions.length - 1 ? (
                <CustomButton
                  text="Next"
                  onClick={handleNext}
                  disabled={answers[currentQuestionIndex] === null}
                />
              ) : (
                <CustomButton
                  text={loading ? "Processing..." : "Submit"}
                  onClick={handleSubmit}
                  disabled={loading || answers[currentQuestionIndex] === null}
                />
              )}
            </Box>
          </>
        )}
      </CardContainer>
    </CustomContainer>
  );
};

export default TestMBTI;
