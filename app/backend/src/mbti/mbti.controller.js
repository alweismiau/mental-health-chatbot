const axios = require("axios");

const getMBTIResult = async (req, res) => {
  try {
    const { answers } = req.body;
    if (!answers || answers.length !== 12) {
      return res.status(400).json({ message: "Invalid answers" });
    }

    // Kirim data ke API Python (Flask)
    const response = await axios.post("http://localhost:5001/mbti-test", { answers });
    
    res.status(200).json({ mbti_result: response.data.mbti_result });
  } catch (error) {
    res.status(500).json({ message: "Error processing MBTI test", error: error.message });
  }
};

module.exports = { getMBTIResult };
