def calculate_mbti(answers):
    scores = {
        "E": 0, "I": 0,
        "S": 0, "N": 0,
        "T": 0, "F": 0,
        "J": 0, "P": 0
    }
    
    for answer in answers:
        if answer in scores:
            scores[answer] += 1

    mbti_result = (
        "E" if scores["E"] > scores["I"] else "I",
        "S" if scores["S"] > scores["N"] else "N",
        "T" if scores["T"] > scores["F"] else "F",
        "J" if scores["J"] > scores["P"] else "P"
    )

    return "".join(mbti_result)
